# -*- coding: utf-8 -*-
# Student ID: 260201044   
def A_decision_tree(Istat):
    if lstat >= 9.7:
        if lstat >= 20:
            if nox >= 0.6:
                return "10"
            else:
                return "17"
        else:
            return "18"
    else:
        if rm < 6.9:
            if age < 88:
                if rm < 6.6:
                    return "23"
                else:
                    return "28"
            else:
                return "36"
        else:
            if rm < 7.4:
                return "34"
            else:
                return "45"

def B_decision_tree(rm):
    if rm < 7.1:
        if lstat >= 15:   
            if nox >= 0.6:
                if lstat >= 20:
                    return "10"
                else:
                    return "15"
            else:
                return "18"
        else:
            if rm < 6.5:
                if lstat >= 9.6:
                    return "21"
                else:
                    return "24"
            else:
                if lstat >= 4.9:
                    return "26"
                else:
                    return "32"
    else:
        if rm < 7.4:
            return "33"
        else:
            return "46"

def C_decision_tree(rm):
    if rm < 6.7:
        if lstat >= 14:
            if crim >= 7:
                return "12"
            else:
                return "17"
        else:
            if lstat >= 9.5:
                return "21"
            else:
                if rad < 7.5:
                    return "24"
                else:
                    return "34"
    else:
        if rm < 7.5:
            if lstat >= 5.5:
                if ptratio >= 19:
                    return "22"
                else:
                    return "31"
            else:
                return "34"
        else:
            return "45"
    
lstat = float(input("Enter lstat: ")) 
rm = float(input("Enter rm: "))

if rm < 6.9 and lstat < 9.7:
    age = float(input("Enter age: "))
if lstat >= 20:
    nox = float(input("Enter nox: "))
if lstat >= 14 and rm < 6.7:
    crim = float(input("Enter crim: "))
if lstat < 9.5 and rm < 6.7:
    rad = float(input("Enter rad: "))
if lstat >= 5.5:
    if rm >= 6.7 and rm < 7.5:
        ptratio = float(input("Enter ptratio: "))

print("\nDecision tree A: " + A_decision_tree(lstat) )
print("Decision tree B: " + B_decision_tree(rm) )
print("Decision tree C: " + C_decision_tree(rm) )
print("Decision forest: " + str( (int(A_decision_tree(lstat)) + int(B_decision_tree(rm)) + int(C_decision_tree(rm)) ) / 3 ))