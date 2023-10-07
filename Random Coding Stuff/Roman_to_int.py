
def romanToInt(s: str) -> int:
    roman_dict ={"I" : 1, "V" : 5, "X" : 10, "L" : 50, "C" : 100, "D" : 500, "M" : 1000}
    
    return_int = 0
    roman_list = s.split()
    for char in roman_list:
        return_int += roman_dict[char] 

    print(return_int)
    return return_int

test = "III"
print(romanToInt(test))

