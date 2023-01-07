input="try_abq.inp"

line=$( grep '**Job name' "$input" | tr -d '\r')
name=$( sed 's/**Job name: //' <<< $line )
echo "$name"

line=$( grep '**Relative densities' "$input" | tr -d '\r')
rd=$( sed 's/**Relative densities://' <<< $line )
IFS=', ' read -r -a reldens <<< $rd
line=$( grep '**Strut radii' "$input" | tr -d '\r' )
rd=$( sed 's/**Strut radii://' <<< $line )
IFS=', ' read -r -a radii <<< $rd

# for element in "${reldens[@]}"; do
#     echo "$element"
# done

# for element in "${radii[@]}"; do
#     echo "$element"
# done


for index in "${!radii[@]}"; do
    newjob="${name}_${index}"
    sed -e "s/_STRUT_RADIUS_PLACEHOLDER_/${radii[$index]}/" \
      -e "s/**Relative densities.*/**Relative density: ${reldens[$index]}/" \
      -e "s/**Strut radii.*/**Strut radius: ${radii[$index]}/" \
      -e "s/**Job name.*/**Job name: ${newjob}/" $input > "abq_working_dir/$newjob.inp"
    # startnum=$( grep -n 'Start header' "$newjob.inp" | cut -d : -f 1 )
    # endnum=$( grep -n 'End header' "$newjob.inp" | cut -d : -f 1 )
    # sed -n "$(($startnum+1)),$(($endnum-1)) p;${endnum} q" "$newjob.inp" >> headers.head
    # echo "----- simulation_transition -----" >> headers.head
done
# echo $line