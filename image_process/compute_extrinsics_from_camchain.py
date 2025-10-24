import argparse
import os
from typing import Dict, Any

import numpy as np
import yaml


def load_camchain(camchain_path: str) -> Dict[str, Any]:
    with open(camchain_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def list_to_matrix(mat_list: list) -> np.ndarray:
    """
    Convert a 4x4 nested list (row-major) to a numpy array (4x4).
    """
    mat = np.array(mat_list, dtype=float)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {mat.shape}")
    return mat


def compute_extrinsics(camchain: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Given camchain with cam0, cam1, cam2 entries where:
      - cam1.T_cn_cnm1 is T_01 (cam0 -> cam1)
      - cam2.T_cn_cnm1 is T_12 (cam1 -> cam2)

    Return extrinsic transforms relative to cam0 (treat cam0 as world):
      - T_00 = I
      - T_01
      - T_02 = T_01 @ T_12
    """
    T_00 = np.eye(4, dtype=float)

    if 'cam1' not in camchain or 'T_cn_cnm1' not in camchain['cam1']:
        raise KeyError("cam1.T_cn_cnm1 not found in camchain")
    if 'cam2' not in camchain or 'T_cn_cnm1' not in camchain['cam2']:
        raise KeyError("cam2.T_cn_cnm1 not found in camchain")

    T_01 = list_to_matrix(camchain['cam1']['T_cn_cnm1'])
    T_12 = list_to_matrix(camchain['cam2']['T_cn_cnm1'])
    T_02 = T_01 @ T_12

    return {
        'cam0': T_00,
        'cam1': T_01,
        'cam2': T_02,
    }


def to_3x4(extrinsic_4x4: np.ndarray) -> np.ndarray:
    return extrinsic_4x4[:3, :4]


def save_extrinsics(output_path: str, extrinsics: Dict[str, np.ndarray]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    serializable = {
        cam: {
            'T_world_to_cam_4x4': extrinsics[cam].tolist(),
            'T_world_to_cam_3x4': to_3x4(extrinsics[cam]).tolist(),
            # Also include inverse if users need camera-to-world (pose)
            'T_cam_to_world_4x4': np.linalg.inv(extrinsics[cam]).tolist(),
        }
        for cam in ['cam0', 'cam1', 'cam2']
    }
    with open(output_path, 'w') as f:
        yaml.safe_dump(serializable, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description='Compute extrinsics for 3 cameras from camchain.yaml')
    parser.add_argument('--camchain', type=str, required=True, help='Path to camchain.yaml')
    parser.add_argument('--out', type=str, default='', help='Output YAML path (default: alongside input as cam_extrinsics.yaml)')
    args = parser.parse_args()

    camchain = load_camchain(args.camchain)
    extrinsics = compute_extrinsics(camchain)

    # Default output next to input
    out_path = args.out
    if not out_path:
        base_dir = os.path.dirname(os.path.abspath(args.camchain))
        out_path = os.path.join(base_dir, 'cam_extrinsics.yaml')

    save_extrinsics(out_path, extrinsics)

    # Pretty print to console
    np.set_printoptions(precision=6, suppress=True)
    print('Computed extrinsics (world = cam0):')
    for cam in ['cam0', 'cam1', 'cam2']:
        print(f"\n{cam} T_world_to_cam (4x4):\n{extrinsics[cam]}")
        print(f"{cam} [R|t] (3x4):\n{to_3x4(extrinsics[cam])}")
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()


