#!/bin/bash

rm -rf ../NII-CU/rgb-t

mkdir ../niicu ../niicu/train ../niicu/train/images ../niicu/train/labels
cp -r ../NII-CU/4-channel/labels/train/* ../niicu/train/labels

mkdir ../niicu/val ../niicu/val/images ../niicu/val/labels
cp -r ../NII-CU/4-channel/labels/val/* ../niicu/val/labels

cp ../configs/niicu.yaml ../niicu