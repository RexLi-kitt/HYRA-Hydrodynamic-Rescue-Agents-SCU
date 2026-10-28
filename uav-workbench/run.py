"""HyRA UAV MVP entry point; no external simulation engine or training needed."""
import argparse
from pathlib import Path
from hyra_uav.core import FloodProcess
from hyra_uav.experiments import OUTPUT, available_scenes, build_fixtures, build_logic_demo, pilot, run_episode, save_run, write_json


def main():
    parser = argparse.ArgumentParser(description="HyRA v2 无人机观测工作台")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-fixtures", help="使用现有求解器生成三类合成开发过程")
    sub.add_parser("build-logic-demo", help="生成不计入洪水证据的人工验收时序")
    serve = sub.add_parser("serve", help="启动本机回放与规则策略工作台")
    serve.add_argument("--port", type=int, default=8767)
    audit = sub.add_parser("audit", help="核查完整洪水导出ZIP")
    audit.add_argument("path", type=Path)
    audit.add_argument("--output", type=Path, default=OUTPUT / "p0-audit.json")
    demo = sub.add_parser("demo", help="保存规则策略演示和观测日志")
    demo.add_argument("--scene", default="rapid")
    demo.add_argument("--method", choices=("R0", "R1", "R2"), default="R2")
    demo.add_argument("--output", type=Path, required=True)
    study = sub.add_parser("pilot", help="每套参数运行27次开发试跑")
    study.add_argument("--output", type=Path, required=True)
    study.add_argument("--durations", type=int, nargs="+", default=[600])
    report = sub.add_parser("report", help="用Matplotlib导出开发比较曲线")
    report.add_argument("pilot", type=Path)
    report.add_argument("--output", type=Path, required=True)
    p3 = sub.add_parser("p3", help="六个新母地形、18条固定回放及P3线性筛查")
    p3.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "serve":
        from hyra_uav.web import serve
        serve(args.port)
    elif args.command == "build-fixtures":
        build_fixtures()
    elif args.command == "build-logic-demo":
        build_logic_demo()
    elif args.command == "audit":
        write_json(args.output, FloodProcess(args.path).audit())
        print(args.output)
    elif args.command == "demo":
        scene = next((s for s in available_scenes() if s["id"] == args.scene), None)
        if scene is None:
            parser.error("场景不存在，请先生成开发过程")
        from hyra_uav.core import Config, Pose
        result = run_episode(scene, args.method, Config(**scene["recommended_config"]) if "recommended_config" in scene else None,
                             Pose(**scene["demo_pose"]) if "demo_pose" in scene else None, scene.get("demo_actions"))
        save_run(result, args.output)
        print(result["summary"])
    elif args.command == "pilot":
        rows = pilot(args.output, tuple(args.durations))
        if any(row["status"] != "completed" for row in rows):
            raise SystemExit("部分运行失败，详见 index.json")
    elif args.command == "report":
        from hyra_uav.reporting import report
        report(args.pilot, args.output)
    elif args.command == "p3":
        from hyra_uav.diagnostic import diagnose
        diagnose(args.output)


if __name__ == "__main__":
    main()
