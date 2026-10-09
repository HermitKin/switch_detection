"""Regression tests for coordinate conventions, matching, and reproducible replay."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from switch_orientation_demo.algorithm import (
    ROOT, Detection, ReplayDetector, clockwise_from_up, direction_name,
    estimate_direction, match_switches, obb_geometry, render_result,
)
from switch_orientation_demo.main import collect_images, run_headless


def box(cx, cy, width, height, class_id=0):
    points = np.array([[cx-width/2,cy-height/2], [cx+width/2,cy-height/2],
                       [cx+width/2,cy+height/2], [cx-width/2,cy+height/2]], np.float32)
    return Detection(class_id, "switch_handle" if class_id==0 else "angle", 0.9, points, 0.0)


class OrientationTests(unittest.TestCase):
    def test_cardinal_angles_and_bins(self):
        for vector, degree, name in [((0,-1),0,"上"), ((1,0),90,"右"), ((0,1),180,"下"), ((-1,0),270,"左")]:
            self.assertEqual(clockwise_from_up(np.array(vector)), degree)
            self.assertEqual(direction_name(degree), name)
        self.assertEqual(direction_name(359), "上")
        self.assertEqual(direction_name(22.49), "上")
        self.assertEqual(direction_name(22.5), "右上")

    def test_corner_order_does_not_change_axis(self):
        points = box(50,50,60,10,1).points
        for candidate in (points, np.roll(points,1,axis=0), points[::-1]):
            _, a, b, length, width = obb_geometry(candidate)
            self.assertAlmostEqual(length,60)
            self.assertAlmostEqual(width,10)
            self.assertAlmostEqual(clockwise_from_up(b-a)%180,90)

    def test_matching_is_one_to_one_and_keeps_orphans(self):
        switches = [box(30,30,35,35), box(100,30,35,35)]
        handles = [box(30,30,20,6,1), box(32,30,20,6,1), box(100,30,20,6,1), box(250,250,20,6,1)]
        matches, orphans = match_switches(switches+handles)
        self.assertEqual(len(matches),2)
        self.assertEqual(len({id(m.angle) for m in matches}),2)
        self.assertEqual(len(orphans),2)

    def test_no_detections_and_no_handle(self):
        image = Image.new("RGB",(100,100),"white")
        _, rows, summary = render_result(image,"blank.jpg",[],True)
        self.assertEqual(rows,[])
        self.assertEqual(summary["switches"],0)
        _, rows, _ = render_result(image,"blank.jpg",[box(50,50,20,20)],True)
        self.assertEqual(rows[0]["direction"],"未检测到angle")
        self.assertFalse(rows[0]["direction_reliable"])

    def test_indicator_is_opposite_lever_end(self):
        image = Image.new("RGB",(120,100),"white")
        estimate = estimate_direction(image,box(40,50,60,60),box(60,50,60,10,1))
        self.assertAlmostEqual(estimate["direction_degrees"],270)
        self.assertEqual(estimate["direction"],"左")
        self.assertTrue(0<=estimate["direction_confidence"]<=1)

    def test_replay_rejects_changed_image_and_settings(self):
        detector = ReplayDetector(ROOT/"samples/predictions.json")
        sample = ROOT/"samples/ds01_remaining_000141.jpg"
        detections,_ = detector.detect(sample,Path("unused"),.25,.55,640,"cpu")
        self.assertEqual(len(detections),18)
        with self.assertRaises(ValueError): detector.detect(sample,Path("unused"),.25,.7,640,"cpu")
        with tempfile.TemporaryDirectory() as temporary:
            changed = Path(temporary)/sample.name
            changed.write_bytes(b"modified")
            with self.assertRaises(ValueError): detector.detect(changed,Path("unused"),.25,.55,640,"cpu")

    def test_empty_folder_and_safe_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError): collect_images(Path(temporary))
        args = argparse.Namespace(input=ROOT/"samples",output=ROOT/"samples/out",replay=ROOT/"samples/predictions.json")
        with self.assertRaises(ValueError): run_headless(args)

    def test_batch_preserves_duplicate_filenames(self):
        sample=ROOT/"samples/ds01_remaining_000229.jpg"
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            for sub in ("a","b"):
                destination=base/"input"/sub/sample.name
                destination.parent.mkdir(parents=True)
                destination.write_bytes(sample.read_bytes())
            args=argparse.Namespace(input=base/"input",output=base/"out",replay=ROOT/"samples/predictions.json",
                                    model=Path("unused"),conf=.25,iou=.55,imgsz=640,device="cpu",show_direction=True)
            self.assertEqual(run_headless(args),0)
            self.assertEqual(len(list((base/"out").rglob("*.jpg"))),2)
            payload=json.loads((base/"out/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"],"replay")
            self.assertEqual({row["image"] for row in payload["objects"]},
                             {f"a/{sample.name}", f"b/{sample.name}"})


if __name__=="__main__": unittest.main()
