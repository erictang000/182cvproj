"""
Ref: https://github.com/Axquaris/cs182project/blob/master/test_submission_torch.py
Add imports here
"""
import pathlib
import sys

#model_path = _____

def main():
    """
    load stuff.
    """
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])
    im_height, im_width = 64, 64
    eval_csv_path = sys.argv[1]

    #model =  ___

    with open('eval_classified.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        eval_dir = sys.argv[1]
        input_path = pathlib.Path(eval_csv_path])
        for line in input_path.open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(',')
            print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            #model specific stuff here
            outputs = model(img)
            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))



if __name__ == '__main__':
    main()
