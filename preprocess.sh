#!/bin/bash
ENTITIES="n2c2"
MAX_LENGTH=128

for ENTITY in $ENTITIES
do
	echo "***** " $ENTITY " Preprocessing Start *****"
	DATA_DIR=../datasets/NER/$ENTITY

	# Replace tab to space
	
	cat $DATA_DIR/devel.tsv | tr '\t' ' ' > $DATA_DIR/devel.txt.tmp
	cat $DATA_DIR/train_dev.tsv | tr '\t' ' ' > $DATA_DIR/train_dev.txt.tmp
	
	echo "Replacing Done"

	# Preprocess for BERT-based models

	python scripts/preprocess.py $DATA_DIR/devel.txt.tmp bert-base-cased $MAX_LENGTH > $DATA_DIR/devel.txt
	python scripts/preprocess.py $DATA_DIR/train_dev.txt.tmp bert-base-cased $MAX_LENGTH > $DATA_DIR/train_dev.txt
	cat $DATA_DIR/train_dev.txt $DATA_DIR/devel.txt  | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_DIR/labels.txt
	echo "***** " $ENTITY " Preprocessing Done *****"
done
