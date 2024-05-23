from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    """
        image/path.tiff x,w,y,h,idx x,w,y,h,idx ...
    """
    labels_dir_template = "/home/frencis/D1/datasets/m3fd/{split}/labels"
    labels_file_template = "/home/frencis/D1/datasets/m3fd/{split}_labels.txt"
    image_path_template = "/home/frencis/D1/datasets/m3fd/{split}/images/{file_name}.tiff"

    img_height, img_width = 768, 1024
    new_img_height, new_img_width = 416, 416

    scale_width = new_img_width / img_width
    scale_height = new_img_height / img_height

    # Convert llvip labels to yolo
    for split in ('train', 'val'):
        labels_path = Path(labels_dir_template.format(split=split))

        labels_txt = ""
        for label_file in tqdm(labels_path.rglob('*.txt'), desc=split):
            text_line = image_path_template.format(
                split=split,
                file_name=label_file.name.replace('.txt', ''),
            )

            # boxes = []

            # Write each line reformatted into the new file
            with open(label_file, 'r') as file:
                for line in file:
                    bb_data = line.strip()

                    idx, x, y, w, h = map(float, bb_data.split())

                    # Convert decimal coordinates to pixels
                    x_pixel = int(x * scale_width * img_width)
                    y_pixel = int(y * scale_height * img_height)
                    w_pixel = int(w * scale_width * img_width)
                    h_pixel = int(h * scale_height * img_height)
                    idx = int(idx)

                    # boxes.append(
                    #     (
                    #         int(x_pixel - (w_pixel/2)), # x_min
                    #         int(y_pixel - (h_pixel/2)), # y_min
                    #         int(x_pixel + (w_pixel/2)), # x_max
                    #         int(y_pixel + (h_pixel/2)), # y_max
                    #     )
                    # )

                    text_line += f" {x_pixel},{w_pixel},{y_pixel},{h_pixel},{idx}"
            labels_txt += f"{text_line}\n"

            # image = Image.open(
            #     image_path_template.format(
            #         split=split,
            #         file_name=label_file.name.replace('.txt', ''),
            #     )
            # ).convert('RGB').resize((416, 416))
            # draw = ImageDraw.Draw(image)
            # for x_min, y_min, x_max, y_max in boxes:
            #     bbox = (x_min, y_min, x_max, y_max)
            #     draw.rectangle(bbox, outline="lime", width=1)
            # image.save(label_file.name.replace('.txt', '.jpg'))
            # break

        # Write extracted labels to the common file
        with open(labels_file_template.format(split=split), 'w') as file:
            file.write(labels_txt)