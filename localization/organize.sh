
array_contains () {
  local array="$1[@]"
  local seeking=$2
  local in=1
  for element in "${!array}"; do
    if [[ $element == $seeking ]]; then
      in=0
      break
    fi
  done
  return $in
}

ids=( $(ls $1 | egrep -e "(additional_[^\.]+)" -o) )
rois=( $(ls $2 | egrep -e "(additional_[^\.]+)" -o) )

type=${1##*/}

for roi in "${rois[@]}"; do
  if array_contains ids $roi; then
    mv "$2/$roi.roi" "$2/$type/"
  fi
done

