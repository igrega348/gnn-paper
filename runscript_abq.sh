#!/bin/bash
#$ -l h_rt=30:00:00
#$ -N gnn_abq_0
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 32
/bin/echo Running on host: `hostname`.
/bin/echo Starting on: `date`

num_processors=32
run_num=0

cdir=$PWD
dname="/scratch/ig348/to_run_gnn_$run_num"
mkdir $dname
tarname="input_files_cat_${run_num}.tar.gz"
cp $tarname $dname
cp ./abq_analyse_parallel.py $dname
cd $dname
echo "Current directory $PWD" 

abq_wdir=$( realpath "abq_working_dir" )
mkdir "${abq_wdir}"
input_dir=$( realpath "input_files" )
processed_dir=$( realpath "processed_data" )
mkdir "${processed_dir}"
processed_tar="processed_data_$run_num.tar.gz"

# make sure we have all files we need
if [[ ! -f abq_analyse_parallel.py ]]; then
	echo "Data extraction script does not exist"
  exit 1
fi
if [[ ! -f "$tarname" ]]; then
	echo "Input files not found"
  exit 1
fi

echo "Extracting input files"
tar -xzf $tarname
if [[ ! -d "$input_dir" ]]; then
	echo "Wrong directory to input files"
  exit 1
fi
echo "Extracted $(ls $input_dir | wc -l) files"


echo "Running python cae extractor"
# the script expects to see the following folders:
#   abq_working_dir, processed_data
# it will look for file
#   simulations_completed.log 
# it will produce file
#   odb_extraction_completed.log
abq2020 cae noGUI=abq_analyse_parallel.py &

if [[ -f "completed.txt" ]]; then
  rm "completed.txt"
fi
fs=$( ls $input_dir | grep .inp )
Ntotal=$( wc -l <<< $fs )
i=0
i_base=0
CUR_JOBS=0
i_check=$(( $i + ${num_processors} - 2 ))
start=`date +%s`

cd "${abq_wdir}"


for f in $fs; do
  if [ $i -ge $i_check ]; then
    CUR_JOBS=$(find . -maxdepth 1 -type f -name '*.lck' -printf x | wc -c)
    check_step=$(( ${num_processors} - 1 - $CUR_JOBS ))
    check_step=$(( 1 > $check_step ? 1 : $check_step ))
    i_check=$(( $i + $check_step ))
  fi
  until [[ $CUR_JOBS < $((${num_processors} - 1)) ]]; do
    sleep 0.1
    CUR_JOBS=$(find . -maxdepth 1 -type f -name '*.lck' -printf x | wc -c)
  done
  
  input_file_path=$( realpath "$input_dir/$f" )

  line=$(grep "**Job name" "${input_file_path}" | tr -d '\r' )
  name=$(sed 's/**Job name: //' <<< $line)

  line=$( grep '**Relative densities' "${input_file_path}" | tr -d '\r')
  rd=$( sed 's/**Relative densities://' <<< $line )
  IFS=', ' read -r -a reldens <<< $rd
  line=$( grep '**Strut radii' "${input_file_path}" | tr -d '\r' )
  rd=$( sed 's/**Strut radii://' <<< $line )
  IFS=', ' read -r -a radii <<< $rd


  for index in "${!radii[@]}"; do
      newjob="${name}_${index}"
      sed -e "s/_STRUT_RADIUS_PLACEHOLDER_/${radii[$index]}/" \
        -e "s/**Relative densities.*/**Relative density: ${reldens[$index]}/" \
        -e "s/**Strut radii.*/**Strut radius: ${radii[$index]}/" \
        -e "s/**Job name.*/**Job name: ${newjob}/" "${input_file_path}" > "$newjob.inp"

      abq2020 job="$newjob" ask_delete=off &
      echo "$newjob" >> ../completed.txt
      i=$(( $i+1 ))

  done
  
  i_base=$(($i_base + 1))
  
  tnow=`date +%s`
  secpernum=$((100*($tnow-$start)/${i_base}))
  estimated=$((($Ntotal-${i_base})*$secpernum/100))
  echo "ibase=${i_base} / $Ntotal, name: $name, estimated time to finish: $estimated seconds"
done

CUR_JOBS=$(find . -maxdepth 1 -type f -name '*.lck' -printf x | wc -c)

until [ $CUR_JOBS -lt 1 ] ; do
  echo "Waiting for completion"
  sleep 1
  CUR_JOBS=$(find . -maxdepth 1 -type f -name '*.lck' -printf x | wc -c)
done

cd ${dname}

echo "Wait for completion of extraction"
until [ $(find . -maxdepth 1 -type f -name 'odb_extraction_completed.log' -printf x | wc -c) -eq 1 ]; do
  echo "Waiting for completion"
  sleep 5
done

ls 

tar -czf $processed_tar "$( basename ${processed_dir} )/"

cp $processed_tar $cdir

cd ..
echo "Removing directory"
rm -r $dname

/bin/echo Finished on: `date`

# EOF
