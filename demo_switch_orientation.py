"""兼容入口：实际实现位于 switch_orientation_demo 包。"""

from switch_orientation_demo.algorithm import *  # noqa: F401,F403
from switch_orientation_demo.main import main


if __name__ == "__main__":
    raise SystemExit(main())
