#!/bin/bash

/usr/bin/curl -s "$1" > input.json;
/usr/bin/curl -s echo "$2" >> active_transfer_learning.py;
/usr/bin/curl -s "$3" > embedding.csv.bz2;
/usr/local/bin/python run_ml.py "$4" "$5";
