#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 18:31:14 2021

@author: user
"""


import detect
import RegisterandTrain


choice=int(input('''Enter your choice : 
                 
                 1. Register new student 
                 2. Mark attendence
                 3. Exit
                 
                 -->'''))
while choice<=3:
    if choice==1:
        RegisterandTrain.new_registration()
        RegisterandTrain.train_model()
        break
    if choice==2:
        detect.detect_student()
        break
    if choice==3:
        break
        
    
        