
declare -a StringArray=("appricot" "black_grapes" "green_lemon" "red_peach" "pear" "small_banana" "avocado" "green_apple" "red_apple" "strawberry" "tomato" "big_banana" "green_grapes" "red_grapes" "yellow_lemon")
 
# Iterate the string array using for loop
for val in ${StringArray[@]}; do
   python setup/write_npy_real_veggies.py $val
done