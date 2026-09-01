import json
import pandas as pd

PAIRS = [
    (
        "pose_6",
        "output_data/poses/kpts3d_00_6_20251215_232147_m_joint_gcvspl.csv",
        "output_data/torque/kpts3d_00_6_20251215_232147_m_joint_gcvspl_torque.csv",
    ),
    (
        "pose_3",
        "output_data/poses/kpts3d_00_3_20251215_232033_m_joint_gcvspl.csv",
        "output_data/torque/kpts3d_00_3_20251215_232033_m_joint_gcvspl_torque.csv",
    ),
]


def main() -> None:
    out = []
    for name, pose_csv, torque_csv in PAIRS:
        df_pose = pd.read_csv(pose_csv, usecols=["frame"])
        df_torque = pd.read_csv(torque_csv, usecols=["frame"])

        missing = sorted(set(df_pose.frame) - set(df_torque.frame))
        extra = sorted(set(df_torque.frame) - set(df_pose.frame))

        out.append(
            {
                "name": name,
                "pose_len": len(df_pose),
                "pose_first": int(df_pose.frame.iloc[0]),
                "pose_last": int(df_pose.frame.iloc[-1]),
                "torque_len": len(df_torque),
                "torque_first": int(df_torque.frame.iloc[0]),
                "torque_last": int(df_torque.frame.iloc[-1]),
                "missing": len(missing),
                "extra": len(extra),
                "missing_example": missing[:5],
                "extra_example": extra[:5],
            }
        )

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
