out = 'decision_tree_rules_clean_best.txt'

with open('decision_tree_rules_clean.txt', 'r') as f:
    lines = f.readlines()

with open(out, 'a') as f:
    for i in range(2):
        for line in lines:
            if 'r101_xc_distance' in line:
                line = line.replace('r101_xc_distance', 'r101_xc_class_' + str(i).zfill(3) + '_distance')
            if 'r101_xc_count1' in line:
                line = line.replace('r101_xc_count1', 'r101_xc_class_' + str(i).zfill(3) + '_count1')
            if 'r101_xc_count8' in line:
                line = line.replace('r101_xc_count8', 'r101_xc_class_' + str(i).zfill(3) + '_count8')
            if 'r50_x4_distance' in line:
                line = line.replace('r50_x4_distance', 'r50_x4_class_' + str(i).zfill(3) + '_distance')
            if 'r50_x4_count1' in line:
                line = line.replace('r50_x4_count1', 'r50_x4_class_' + str(i).zfill(3) + '_count1')
            if 'r50_x4_count8' in line:
                line = line.replace('r50_x4_count8', 'r50_x4_class_' + str(i).zfill(3) + '_count8') 
            if 'r50_x4_count7' in line:
                line = line.replace('r50_x4_count7', 'r50_x4_class_' + str(i).zfill(3) + '_count7')
            if 'r50_xc_distance' in line:
                line = line.replace('r50_xc_distance', 'r50_xc_class_' + str(i).zfill(3) + '_distance')
            if 'r50_xc_count1' in line:
                line = line.replace('r50_xc_count1', 'r50_xc_class_' + str(i).zfill(3) + '_count1')
            if 'r50_xc_count8' in line:
                line = line.replace('r50_xc_count8', 'r50_xc_class_' + str(i).zfill(3) + '_count8')     
            if 'r50_xc_count7' in line:
                line = line.replace('r50_xc_count7', 'r50_xc_class_' + str(i).zfill(3) + '_count7')
            if 'r101_x4_distance' in line:
                line = line.replace('r101_x4_distance', 'r101_x4_class_' + str(i).zfill(3) + '_distance')
            if 'r101_x4_count1' in line:
                line = line.replace('r101_x4_count1', 'r101_x4_class_' + str(i).zfill(3) + '_count1')
            if 'r101_x4_count8' in line:
                line = line.replace('r101_x4_count8', 'r101_x4_class_' + str(i).zfill(3) + '_count8')
            if 'd169_x4_distance' in line:
                line = line.replace('d169_x4_distance', 'd169_x4_class_' + str(i).zfill(3) + '_distance')
            if 'd169_x4_count1' in line:
                line = line.replace('d169_x4_count1', 'd169_x4_class_' + str(i).zfill(3) + '_count1')
            if 'd169_x4_count8' in line:
                line = line.replace('d169_x4_count8', 'd169_x4_class_' + str(i).zfill(3) + '_count8')
            if 'd169_xc_distance' in line:
                line = line.replace('d169_xc_distance', 'd169_xc_class_' + str(i).zfill(3) + '_distance')
            if 'd169_xc_count1' in line:
                line = line.replace('d169_xc_count1', 'd169_xc_class_' + str(i).zfill(3) + '_count1')
            if 'd169_xc_count8' in line:
                line = line.replace('d169_xc_count8', 'd169_xc_class_' + str(i).zfill(3) + '_count8')
            if 'transform_xc_count1' in line:
                line = line.replace('transform_xc_count1', 'transform_xc_class_' + str(i).zfill(3) + '_count1')
            if 'transform_xc_count8' in line:
                line = line.replace('transform_xc_count8', 'transform_xc_class_' + str(i).zfill(3) + '_count8')
            if 'transform_x4_count8' in line:
                line = line.replace('transform_x4_count8', 'transform_x4_class_' + str(i).zfill(3) + '_count8')
            if 'transform_x4_count1' in line:
                line = line.replace('transform_x4_count1', 'transform_x4_class_' + str(i).zfill(3) + '_count1')
            if 'transform_x4_distance' in line:
                line = line.replace('transform_x4_distance', 'transform_x4_class_' + str(i).zfill(3) + '_distance')
            if 'CCT_xc_count1' in line:
                line = line.replace('CCT_xc_count1', 'CCT_xc_class_' + str(i).zfill(3) + '_count1')
            if 'CCT_xc_count8' in line:
                line = line.replace('CCT_xc_count8', 'CCT_xc_class_' + str(i).zfill(3) + '_count8')   
            if 'CCT_xc_count7' in line:
                line = line.replace('CCT_xc_count7', 'CCT_xc_class_' + str(i).zfill(3) + '_count7')
            if 'CCT_x4_count8' in line:
                line = line.replace('CCT_x4_count8', 'CCT_x4_class_' + str(i).zfill(3) + '_count8')  
            if 'CCT_x4_count7' in line:
                line = line.replace('CCT_x4_count7', 'CCT_x4_class_' + str(i).zfill(3) + '_count7')
            if 'CCT_x4_count1' in line:
                line = line.replace('CCT_x4_count1', 'CCT_x4_class_' + str(i).zfill(3) + '_count1')
            if 'CCT_x4_distance' in line:
                line = line.replace('CCT_x4_distance', 'CCT_x4_class_' + str(i).zfill(3) + '_distance')   
            if 'CCT_xc_distance' in line:
                line = line.replace('CCT_xc_distance', 'CCT_xc_class_' + str(i).zfill(3) + '_distance')
            if 'r50_xc_fangcha' in line:
                line = line.replace('r50_xc_fangcha', 'r50_xc_class_' + str(i).zfill(3) + '_fangcha')   
            if 'r50_xc_xs8' in line:
                line = line.replace('r50_xc_xs8', 'r50_xc_class_' + str(i).zfill(3) + '_xs8')  
            if 'r50_xc_xs5' in line:
                line = line.replace('r50_xc_xs5', 'r50_xc_class_' + str(i).zfill(3) + '_xs5')   
            if 'r50_xc_xs3' in line:
                line = line.replace('r50_xc_xs3', 'r50_xc_class_' + str(i).zfill(3) + '_xs3') 
            if 'r50_xc_xs1' in line:
                line = line.replace('r50_xc_xs1', 'r50_xc_class_' + str(i).zfill(3) + '_xs1') 
            if 'r50_xc_padcount1' in line:
                line = line.replace('r50_xc_padcount1', 'r50_xc_class_' + str(i).zfill(3) + '_padcount1')  
            if 'r50_xc_padcount8' in line:
                line = line.replace('r50_xc_padcount8', 'r50_xc_class_' + str(i).zfill(3) + '_padcount8') 
            if 'r50_x4_padcount1' in line:
                line = line.replace('r50_x4_padcount1', 'r50_x4_class_' + str(i).zfill(3) + '_padcount1')  
            if 'r50_x4_padcount8' in line:
                line = line.replace('r50_x4_padcount8', 'r50_x4_class_' + str(i).zfill(3) + '_padcount8')  
            if 'r50_xc_paddingdis' in line:
                line = line.replace('r50_xc_paddingdis', 'r50_xc_class_' + str(i).zfill(3) + '_paddingdis')   
            if 'r50_x4_paddingdis' in line:
                line = line.replace('r50_x4_paddingdis', 'r50_x4_class_' + str(i).zfill(3) + '_paddingdis')   
            if 'r50_xc_xsdistance' in line:
                line = line.replace('r50_xc_xsdistance', 'r50_xc_class_' + str(i).zfill(3) + '_xsdistance')
            if 'r50_xc_all5' in line:
                line = line.replace('r50_xc_all5', 'r50_xc_class_' + str(i).zfill(3) + '_all5')
            if 'r50_xc_all3' in line:
                line = line.replace('r50_xc_all3', 'r50_xc_class_' + str(i).zfill(3) + '_all3')
            if 'M' in line:
                line = line.replace('M', 'M' + str(i))
            if 'U' in line:
                line = line.replace('U', 'U' + str(i))
                
            f.write(line)
