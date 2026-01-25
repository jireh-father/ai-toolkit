import os
import json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/image")
async def get_image(path: str = Query(...)):
    """Serve local image file"""
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({"error": "File not found"}, status_code=404)


@app.post("/api/scan-folders")
async def scan_folders(request: Request):
    """Scan 3 folders and create/load mapping JSON"""
    data = await request.json()
    folder1 = data.get("folder1", "")  # input
    folder2 = data.get("folder2", "")  # output
    folder3 = data.get("folder3", "")  # reference

    # Validate folders
    for folder in [folder1, folder2, folder3]:
        if not folder or not os.path.isdir(folder):
            return JSONResponse({"error": f"Invalid folder path: {folder}"}, status_code=400)

    # Get lowest subfolder names
    name1 = Path(folder1).name
    name2 = Path(folder2).name
    name3 = Path(folder3).name

    # Get image files from each folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def get_image_files(folder):
        files = {}
        for f in os.listdir(folder):
            ext = os.path.splitext(f)[1].lower()
            if ext in image_extensions:
                name_without_ext = os.path.splitext(f)[0]
                files[name_without_ext] = f
        return files

    files1 = get_image_files(folder1)
    files2 = get_image_files(folder2)
    files3 = get_image_files(folder3)

    # Find common filenames (by name without extension)
    common_names = set(files1.keys()) & set(files2.keys()) & set(files3.keys())
    common_names = sorted(common_names)

    # Create JSON filename
    json_filename = f"{name1}_{name2}_{name3}_{len(common_names)}.json"
    json_path = os.path.join(folder1, json_filename)

    # Check if JSON already exists
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)

        # Handle both old format (list) and new format (dict with folders)
        if isinstance(existing_data, list):
            items = existing_data
        else:
            items = existing_data.get("items", [])

        return JSONResponse({
            "json_path": json_path,
            "json_filename": json_filename,
            "data": items,
            "folders": {
                "folder1": folder1,
                "folder2": folder2,
                "folder3": folder3
            },
            "loaded_existing": True
        })

    # Create new mapping
    mappings = []
    for name in common_names:
        mappings.append({
            "id": name,
            "image1": os.path.join(folder1, files1[name]),
            "image2": os.path.join(folder2, files2[name]),
            "image3": os.path.join(folder3, files3[name]),
            "status": "none"  # "approved", "deleted", or "none"
        })

    # Save JSON with folder paths
    save_data = {
        "folders": {
            "folder1": folder1,
            "folder2": folder2,
            "folder3": folder3
        },
        "items": mappings
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    return JSONResponse({
        "json_path": json_path,
        "json_filename": json_filename,
        "data": mappings,
        "folders": {
            "folder1": folder1,
            "folder2": folder2,
            "folder3": folder3
        },
        "loaded_existing": False
    })


@app.post("/api/update-item")
async def update_item(request: Request):
    """Update a single item's status in the JSON file"""
    data = await request.json()
    json_path = data.get("json_path", "")
    item_id = data.get("id", "")
    status = data.get("status", "none")  # "approved", "deleted", or "none"

    if not json_path or not os.path.exists(json_path):
        return JSONResponse({"error": "JSON file not found"}, status_code=400)

    # Load, update, save
    with open(json_path, 'r', encoding='utf-8') as f:
        file_data = json.load(f)

    # Handle both old format (list) and new format (dict with folders)
    if isinstance(file_data, list):
        items = file_data
        for item in items:
            if item["id"] == item_id:
                item["status"] = status
                # Remove old fields if present
                item.pop("approved", None)
                item.pop("deleted", None)
                break
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    else:
        items = file_data.get("items", [])
        for item in items:
            if item["id"] == item_id:
                item["status"] = status
                # Remove old fields if present
                item.pop("approved", None)
                item.pop("deleted", None)
                break
        file_data["items"] = items
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)

    return JSONResponse({"success": True})


@app.get("/api/load-label-file")
async def load_label_file(path: str = Query(...)):
    """Load existing label JSON file"""
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    with open(path, 'r', encoding='utf-8') as f:
        file_data = json.load(f)

    # Handle both old format (list) and new format (dict with folders)
    if isinstance(file_data, list):
        return JSONResponse({
            "data": file_data,
            "folders": None
        })
    else:
        return JSONResponse({
            "data": file_data.get("items", []),
            "folders": file_data.get("folders", None)
        })


@app.post("/api/apply-approved")
async def apply_approved(request: Request):
    """Copy approved files to a new location"""
    import shutil

    data = await request.json()
    json_path = data.get("json_path", "")
    target_path = data.get("target_path", "")

    if not json_path or not os.path.exists(json_path):
        return JSONResponse({"error": "JSON file not found"}, status_code=400)

    if not target_path:
        return JSONResponse({"error": "Target path not specified"}, status_code=400)

    # Create target directory if not exists
    os.makedirs(target_path, exist_ok=True)

    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        file_data = json.load(f)

    if isinstance(file_data, list):
        items = file_data
        folders = None
    else:
        items = file_data.get("items", [])
        folders = file_data.get("folders", {})

    if not folders:
        return JSONResponse({"error": "Folder information not found in label file"}, status_code=400)

    # Get folder names
    folder1_name = Path(folders.get("folder1", "")).name
    folder2_name = Path(folders.get("folder2", "")).name
    folder3_name = Path(folders.get("folder3", "")).name

    # Create target folders
    target_folder1 = os.path.join(target_path, folder1_name)
    target_folder2 = os.path.join(target_path, folder2_name)
    target_folder3 = os.path.join(target_path, folder3_name)

    os.makedirs(target_folder1, exist_ok=True)
    os.makedirs(target_folder2, exist_ok=True)
    os.makedirs(target_folder3, exist_ok=True)

    # Copy approved files
    copied_count = 0
    for item in items:
        if item.get("status") == "approved":
            try:
                # Copy image1
                src1 = item.get("image1", "")
                if src1 and os.path.exists(src1):
                    dst1 = os.path.join(target_folder1, os.path.basename(src1))
                    shutil.copy2(src1, dst1)

                # Copy image2
                src2 = item.get("image2", "")
                if src2 and os.path.exists(src2):
                    dst2 = os.path.join(target_folder2, os.path.basename(src2))
                    shutil.copy2(src2, dst2)

                # Copy image3
                src3 = item.get("image3", "")
                if src3 and os.path.exists(src3):
                    dst3 = os.path.join(target_folder3, os.path.basename(src3))
                    shutil.copy2(src3, dst3)

                copied_count += 1
            except Exception as e:
                print(f"Error copying {item.get('id')}: {e}")
                continue

    return JSONResponse({
        "success": True,
        "copied_count": copied_count,
        "target_folders": {
            "folder1": target_folder1,
            "folder2": target_folder2,
            "folder3": target_folder3
        }
    })


@app.post("/api/open-in-explorer")
async def open_in_explorer(request: Request):
    """Open file in Windows Explorer"""
    import subprocess
    import platform

    data = await request.json()
    file_path = data.get("path", "")

    if not file_path or not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=400)

    try:
        if platform.system() == "Windows":
            # Select file in Explorer
            subprocess.run(['explorer', '/select,', file_path], check=False)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(['open', '-R', file_path], check=False)
        else:  # Linux
            folder_path = os.path.dirname(file_path)
            subprocess.run(['xdg-open', folder_path], check=False)

        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/browse")
async def browse_directory(path: str = Query("")):
    """Browse local directory for file/folder selection"""
    # Default to common starting points
    if not path:
        if os.name == 'nt':  # Windows
            drives = []
            for letter in 'CDEFGHIJKLMNOPQRSTUVWXYZ':
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    drives.append({"name": drive, "path": drive, "type": "drive"})
            return JSONResponse({"items": drives, "current_path": "", "parent_path": None})
        else:  # Linux/Mac
            path = os.path.expanduser("~")

    # Normalize path
    path = os.path.normpath(path)

    if not os.path.exists(path):
        return JSONResponse({"error": "Path not found"}, status_code=400)

    if not os.path.isdir(path):
        return JSONResponse({"error": "Not a directory"}, status_code=400)

    items = []
    try:
        for entry in os.scandir(path):
            try:
                item = {
                    "name": entry.name,
                    "path": entry.path,
                    "type": "folder" if entry.is_dir() else "file"
                }
                # For files, add extension info
                if not entry.is_dir():
                    item["ext"] = os.path.splitext(entry.name)[1].lower()
                items.append(item)
            except PermissionError:
                continue
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)

    # Sort: folders first, then files, both alphabetically
    items.sort(key=lambda x: (0 if x["type"] == "folder" else 1, x["name"].lower()))

    # Get parent path
    parent_path = os.path.dirname(path)
    if parent_path == path:  # Root directory
        parent_path = None if os.name != 'nt' else ""

    return JSONResponse({
        "items": items,
        "current_path": path,
        "parent_path": parent_path
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
