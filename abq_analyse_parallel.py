# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import os
import visualization
import json

def parse_input_script(fn):
    with open(fn, 'r') as fin:
        lines = fin.readlines()
    lines = [line.rstrip() for line in lines]
    for i_line, line in enumerate(lines):
        if 'Start header' in line:
            start_line = i_line + 1
        elif 'End header' in line:
            end_line = i_line
            break
    lines = lines[start_line:end_line]
    odict = dict()
    for line in lines:
        fields = line.lstrip('*').split(':')
        key = fields[0]
        val = fields[1].lstrip()
        odict[key] = val
    return odict

def data_extractor():
    if not os.path.isdir('./processed_data'):
        os.mkdir('./processed_data')
    
    input_files = os.listdir('./input_files')
    input_files = [ f for f in input_files if f.endswith('.inp') ]
    NUM_inp_files = len(input_files)
    
    processed = []
    count = 0
    
    while len(processed) < NUM_inp_files:
        output_files = os.listdir('./outputs')
        odb_files = [ (os.path.splitext(f)[0]) for f in output_files if f.endswith('.odb') ]
        lck_files = [ (os.path.splitext(f)[0]) for f in output_files if f.endswith('.lck') ]
        odb_ready = set(odb_files) - set(lck_files)
        to_process = list( odb_ready - set(processed) )
        
        while len(to_process) > 0:
            jobname = to_process.pop()
            processed.append(jobname)

            ODBNAME = './outputs/{}.odb'.format(jobname)
            FILEOUTPUT1 = './processed_data/{}.json'.format(jobname)
            
            # Read the header from input script
            fn = os.path.join('./input_files', '{}.inp'.format(jobname))
            odict = parse_input_script(fn)

            ###############################################################################################################
            o1 = session.openOdb(name=ODBNAME, readOnly=True)
            session.viewports['Viewport: 1'].setValues(displayedObject=o1)    
            odb = session.odbs[ODBNAME]
            assembly = odb.rootAssembly
            #print >> sys.__stdout__, 'odb steps {}, size {}'.format(list(odb.steps.keys()), len(odb.steps.keys()))
            if len(odb.steps.keys())<1:
                print >> sys.__stdout__, 'i = {} failed. Not enough steps'.format(str(jobname))
            else:
                for step_name in ['Load-REF1-dof1','Load-REF1-dof2','Load-REF1-dof3','Load-REF2-dof1','Load-REF2-dof2','Load-REF2-dof3']:
                    odict[step_name] = dict()
                    if len(odb.steps['{}'.format(step_name)].frames)<2: continue
                    frame = odb.steps['{}'.format(step_name)].frames[-1]
                    allFields = frame.fieldOutputs
                    force = allFields['RF']
                    displacement = allFields['U']
                    for ref_pt in [1,2]:
                        refpt = assembly.nodeSets['REF{}'.format(ref_pt)]
                        ref_pt_u = displacement.getSubset(region=refpt)
                        ref_pt_rf = force.getSubset(region=refpt)
                        for i_deg in [1,2,3]:
                            odict[step_name]["REF{}, RF{}".format(ref_pt, i_deg)] = float(ref_pt_rf.values[0].data[i_deg-1])
                        for i_deg in [1,2,3]:
                            odict[step_name]["REF{}, U{}".format(ref_pt, i_deg)] = float(ref_pt_u.values[0].data[i_deg-1])
                if len(odict.keys())<1: 
                    print >> sys.__stdout__, 'i = {} failed. No keys in dictionary'.format(str(jobname))
                    continue
            with open(FILEOUTPUT1,'w') as fout:
                json.dump(odict, fout, indent=4)
            count += 1
            print >> sys.__stdout__, 'i = {}'.format(str(count))
            odb.close()
            
    with open('completed.log', 'w') as fout:
        fout.write('')
        
###########################################################################
data_extractor()

