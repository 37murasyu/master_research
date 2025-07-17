# pylint: disable=no-member
import cv2 as cv
import argparse
import textwrap
import platform

try:
    import wmi  # optional, Windows device info
except ImportError:  # pragma: no cover
    wmi = None


def enumerate_wmi_devices():
    devices = {}
    if wmi is None or platform.system() != 'Windows':
        return devices
    try:
        c = wmi.WMI()
        for cam in c.Win32_PnPEntity():
            name = getattr(cam, 'Name', '') or ''
            if any(k in name.lower() for k in ['camera', 'webcam', 'integrated', 'imaging']):
                devices[name] = cam.DeviceID
    except Exception:  # noqa: E722
        pass
    return devices


def probe_indices(max_index: int, width: int | None, height: int | None, warmup: int = 3):
    results = []
    for idx in range(max_index + 1):
        cap = cv.VideoCapture(idx, cv.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        if width:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue
        # warmup frames (some cameras output first frame black)
        for _ in range(warmup - 1):
            cap.read()
        h, w = frame.shape[:2]
        results.append({
            'index': idx,
            'detected_width': w,
            'detected_height': h,
            'fourcc': int(cap.get(cv.CAP_PROP_FOURCC)),
            'fps': cap.get(cv.CAP_PROP_FPS),
        })
        cap.release()
    return results


def format_fourcc(code: int) -> str:
    if code == 0:
        return '----'
    return ''.join([chr((code >> 8 * i) & 0xFF) for i in range(4)])


def main():
    parser = argparse.ArgumentParser(
        description='接続されている利用可能カメラの index を列挙',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
        使用例:
        python list_cameras.py            # デフォルト 0..5 を探査
        python list_cameras.py -m 10      # 0..10 を探査
        python list_cameras.py -r 1280x720
        '''))
    parser.add_argument('-m', '--max-index', type=int, default=5, help='走査する最大 index (0..max)')
    parser.add_argument('-r', '--resolution', type=str, default=None, help='希望解像度 WxH 例: 1280x720')
    args = parser.parse_args()

    width = height = None
    if args.resolution:
        if 'x' in args.resolution:
            try:
                width, height = map(int, args.resolution.lower().split('x'))
            except ValueError:
                print('解像度指定が不正です。例: 1280x720')
        else:
            print('解像度指定は WxH 形式で入力してください')

    print(f'== カメラ走査開始: 0..{args.max_index} ==')
    wmi_devices = enumerate_wmi_devices()
    if wmi_devices:
        print(f'WMI で候補デバイス {len(wmi_devices)} 件検出:')
        for name, devid in wmi_devices.items():
            print('  -', name, '|', devid)
    else:
        print('WMI からの追加デバイス名情報はありません (権限/非Windows/未インストールの可能性)。')

    probed = probe_indices(args.max_index, width, height)
    if not probed:
        print('利用可能なカメラは見つかりませんでした。')
        return
    print('\n== 利用可能カメラインデックス ==')
    for info in probed:
        print(f"index {info['index']}: {info['detected_width']}x{info['detected_height']} FOURCC={format_fourcc(info['fourcc'])} FPS~{info['fps']:.1f}")

    print('\n最初に使いたい候補:')
    print('  内蔵カメラ → 通常は最も低い index (例: 0)')
    print('  外部USBを避けたい → 列挙結果から内蔵らしい名称の index を指定')

if __name__ == '__main__':
    main()
