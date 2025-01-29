# -*- coding: utf-8 -*-
# Student ID: 260201044    
import sys

outlook = sys.argv[1]
temperature = sys.argv[2]
humidity = sys.argv[3] 
windy = sys.argv[4]

no_decision_pool = ["a1B","a1A","b1A","b2A"]
yes_decision_pool = ["a2B","a2A","b1B","b2B"]
decision=""

if outlook == "overcast":
    play = "Yes"
else:
    if outlook == "sunny":
        decision += "a"
    else:
        decision += "b"
    if humidity == "high":
        decision += "1"
    else:
        decision += "2"
    if windy == "True":
        decision += "A"
    else:
        decision += "B"
        
if decision in no_decision_pool:
    play = "No"
else:
    play = "Yes"        

print(play)