import glob, os, shutil

folder = '/home/hiiko/Videos/Try'

for file_path in glob.glob(os.path.join(folder, '*.*')):
    new_dir = file_path.rsplit('.', 1)[0]
    try:
        os.mkdir(os.path.join(folder, new_dir))
    except:
        # Handle the case where the target dir already exist.
        pass
    shutil.move(file_path, os.path.join(new_dir, os.path.basename(file_path)))
