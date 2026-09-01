param(
    [string[]]$Datasets = @(
        "2_20250925_162436",
        "3_20250925_154950",
        "3_20250925_155122",
        "4_20250925_114613",
        "5_20250925_131537",
        "5_20250925_131643",
        "5_20250925_133228",
        "6_20250925_180723",
        "7_20250925_184912",
        "8_20250925_192700"
    ),
    # default: denser sampling (mapper failed to find initial pair at step=5)
    [int]$Step = 2
)

$ErrorActionPreference = 'Stop'
$root   = "C:/Users/villa/Desktop/master_Research/MAINCODE"
$colmap = "C:/Users/villa/vcpkg/installed/x64-windows/tools/colmap/colmap.exe"
$py     = "C:/Users/villa/venv312/Scripts/python.exe"

foreach ($run in $Datasets) {
    Write-Host "=== Processing $run ===" -ForegroundColor Cyan

    $img = "$root/output_data/colmap_frames/$run"
    $ws  = "$root/output_data/colmap_workspace/$run"

    if (Test-Path $ws) { Remove-Item -Recurse -Force $ws }
    New-Item -ItemType Directory -Force -Path "$img/cam0","$img/cam1","$ws","$ws/sparse" | Out-Null

    $cam0 = "C:/Users/villa/Desktop/master_Research/cameras_raw/$run/cam0_${run}.mp4"
    $cam1 = "C:/Users/villa/Desktop/master_Research/cameras_raw/$run/cam1_${run}.mp4"
    if (-not (Test-Path $cam0) -or -not (Test-Path $cam1)) {
        Write-Warning "Skipping ${run}: cam0 or cam1 mp4 not found"
        continue
    }

    # 1) Extract frames (step sampling via select=not(mod(n,Step)))
    ffmpeg -y -i $cam0 -vf "select=not(mod(n\,$Step))" -vsync vfr "$img/cam0/f%06d.jpg" | Out-Null
    ffmpeg -y -i $cam1 -vf "select=not(mod(n\,$Step))" -vsync vfr "$img/cam1/f%06d.jpg" | Out-Null

    # 2) COLMAP pipeline
    $featureArgs = @(
        "--database_path", "$ws/database.db",
        "--image_path", "$img",
        "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera", 1
    )

    & $colmap feature_extractor @featureArgs

    # Duplicate camera for cam1 and reassign images based on folder prefix
    $sqliteFix = @"
import sqlite3
from pathlib import Path

db = Path(r'$ws/database.db')
conn = sqlite3.connect(db)
cur = conn.cursor()
# Clear rig-related tables; they appear when single_camera is used and break mapper
cur.execute('DELETE FROM rig_sensors')
cur.execute('DELETE FROM frames')
cur.execute('DELETE FROM frame_data')
cur.execute('DELETE FROM rigs')
cams = cur.execute('SELECT camera_id, model, width, height, params, prior_focal_length FROM cameras ORDER BY camera_id').fetchall()
if len(cams) == 1:
    cam0 = cams[0]
    new_id = cam0[0] + 1
    cur.execute('INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, ?)',
                (new_id, cam0[1], cam0[2], cam0[3], cam0[4], cam0[5]))
    cam_ids = [cam0[0], new_id]
else:
    cam_ids = [row[0] for row in cams[:2]]

cur.execute("UPDATE images SET camera_id = ? WHERE name LIKE 'cam1/%'", (cam_ids[1],))
conn.commit()
print('cams', cam_ids)
print('sample', cur.execute('SELECT name, camera_id FROM images LIMIT 5').fetchall())
print('counts', cur.execute('SELECT camera_id, COUNT(*) FROM images GROUP BY camera_id').fetchall())
conn.close()
"@
    $sqliteFix | & $py -

    & $colmap sequential_matcher `
        --database_path "$ws/database.db" `
        --SequentialMatching.loop 1

    & $colmap exhaustive_matcher `
        --database_path "$ws/database.db"

    $mapperArgs = @(
        "--database_path", "$ws/database.db",
        "--image_path", "$img",
        "--output_path", "$ws/sparse",
        # Relaxed/robust settings for low-texture, human-only scenes
        "--Mapper.abs_pose_min_num_inliers", "15",          # default 30
        "--Mapper.init_min_num_inliers", "40",              # default 100
        "--Mapper.min_model_size", "10",                    # build even small models
        "--Mapper.filter_max_reproj_error", "8.0",          # allow higher reprojection error early
        "--Mapper.init_min_tri_angle", "2.0",               # default 4.0; easier initialization
        "--Mapper.ba_global_max_refinements", "5",          # keep BA iterations modest
        "--Mapper.ba_global_max_refinement_change", "1.0e-6"
    )

    & $colmap mapper @mapperArgs

    & $colmap model_converter `
        --input_path "$ws/sparse/0" `
        --output_path "$ws/sparse/0_text" `
        --output_type TXT

    # 3) Export to dat
    & $py "$root/export_colmap_to_dat.py" `
        --model "$ws/sparse/0_text" `
        --out "$root/output_data/calib_selfcal/$run"
}
