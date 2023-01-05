# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
import __main__
import os
import visualization
import json
from collections import OrderedDict

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
    odict = OrderedDict()
    for line in lines:
        fields = line.lstrip('*').split(':')
        key = fields[0]
        val = fields[1].lstrip()
        odict[key] = val
    return odict

def data_extractor():
    wdir = './abq_working_dir'

    if not os.path.isdir('./processed_data'):
        os.mkdir('./processed_data')
    
    to_process = []
    processed = []
    simulations_finished = False
    count = 0
    
    while (not simulations_finished) or (len(to_process)>0):
        if os.path.exists('./simulations_completed.log'):
            simulations_finished = True

        wdir_files = os.listdir(wdir)
       
        odb_names = [ (os.path.splitext(f)[0]) for f in wdir_files if f.endswith('.odb') ]
        lck_names = [ (os.path.splitext(f)[0]) for f in wdir_files if f.endswith('.lck') ]
        odb_ready = set(odb_names) - set(lck_names)
        to_process = list( odb_ready - set(processed) )
        
        while len(to_process) > 0:
            jobname = to_process.pop()
            processed.append(jobname)

            ODBNAME = os.path.join(wdir, '{}.odb'.format(jobname))
            FILEOUTPUT1 = os.path.join('./processed_data', '{}.json'.format(jobname))
            
            # Read the header from input script
            fn = os.path.join(wdir, '{}.inp'.format(jobname))
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
            
    with open('odb_extraction_completed.log', 'w') as fout:
        fout.write('')
        
###########################################################################
data_extractor()

