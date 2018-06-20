#!/bin/sh

label=no_pretrain

for i in `seq 3 20`
do
  number=$(printf "%02d" $i)
  target=./logs/$label/$number
  mkdir -p $target
  rm -f ./results/*
  rm -f ./parameters/*

  python test.py > ./results/log.txt
  # echo ${target}

  cp -r ./results/ $target/
  cp -r ./parameters/ $target/
done
