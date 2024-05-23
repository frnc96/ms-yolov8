from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    """
        image/path.tiff x,w,y,h,idx x,w,y,h,idx ...
    """
    labels_dir_template = "/home/frencis/D1/datasets/niicu/{split}/labels"
    labels_file_template = "/home/frencis/D1/datasets/niicu/{split}_labels.txt"
    image_path_template = "/home/frencis/D1/datasets/niicu/{split}/images/{file_name}.tiff"

    img_height, img_width = 1980, 2706

    # Convert llvip labels to yolo
    for split in ('train', 'val'):
        labels_path = Path(labels_dir_template.format(split=split))

        labels_txt = ""
        for label_file in tqdm(labels_path.rglob('*.txt'), desc=split):
            text_line = image_path_template.format(
                split=split,
                file_name=label_file.name.replace('.txt', ''),
            )

            # Write each line reformatted into the new file
            with open(label_file, 'r') as file:
                for line in file:
                    bb_data = line.strip()

                    idx, x, y, w, h = map(float, bb_data.split())

                    # Convert decimal coordinates to pixels
                    x_pixel = int(x * img_width)
                    y_pixel = int(y * img_height)
                    w_pixel = int(w * img_width)
                    h_pixel = int(h * img_height)
                    idx = int(idx)

                    text_line += f" {x_pixel},{w_pixel},{y_pixel},{h_pixel},{idx}"

            labels_txt += f"{text_line}\n"

        # Write extracted labels to the common file
        with open(labels_file_template.format(split=split), 'w') as file:
            file.write(labels_txt)