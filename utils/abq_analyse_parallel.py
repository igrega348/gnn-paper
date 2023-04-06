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
        val = ':'.join(fields[1:]).lstrip()
        odict[key] = val
    return odict

def data_extractor():
    with open('abq_extract_log.log', 'w') as log_file, open('results_table.csv', 'w') as results_file:
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
                relative_densities = odict['Relative densities'].split(',')
                relative_densities = [float(rd) for rd in relative_densities]
                strut_radii = odict['Strut radii'].split(',')
                strut_radii = [float(sr) for sr in strut_radii]

                ###############################################################################################################
                o1 = session.openOdb(name=ODBNAME, readOnly=True)
                session.viewports['Viewport: 1'].setValues(displayedObject=o1)    
                odb = session.odbs[ODBNAME]
                assembly = odb.rootAssembly
                # 
                loading_steps = ['Load-REF1-dof1','Load-REF1-dof2','Load-REF1-dof3','Load-REF2-dof1','Load-REF2-dof2','Load-REF2-dof3']
                # get instances
                instance_nums = sorted(list(set(
                    [int(x.split('-')[0].lstrip('INST')) for x in assembly.instances.keys()]
                )))
                instances_dict = { num:OrderedDict() for num in instance_nums }

                for i_instance, rd in enumerate(relative_densities):
                    strut_radius = odb.profiles[odb.sections[assembly.instances['INST{}-LAT'.format(i_instance)].sectionAssignments[0].sectionName].profile].r
                    assert abs(strut_radius - strut_radii[i_instance]) < 1e-5
                    instances_dict[i_instance].update({
                        'Relative density':rd,
                        'Strut radius':strut_radius
                    })
                    for step_name in loading_steps:
                        instances_dict[i_instance][step_name] = {}

                #print >> sys.__stdout__, 'odb steps {}, size {}'.format(list(odb.steps.keys()), len(odb.steps.keys()))
                if len(odb.steps.keys())<1:
                    print >> log_file, 'i = {} failed. Not enough steps'.format(str(jobname))
                else:
                    for step_name in loading_steps:
                        
                        if len(odb.steps['{}'.format(step_name)].frames)<2: continue
                        frame = odb.steps['{}'.format(step_name)].frames[-1]
                        allFields = frame.fieldOutputs
                        force = allFields['RF']
                        displacement = allFields['U']
                        for i_instance in instance_nums:
                            for ref_pt in [1,2]:
                                refpt = assembly.nodeSets['INST{}-REF{}'.format(i_instance, ref_pt)]
                                ref_pt_u = displacement.getSubset(region=refpt)
                                ref_pt_rf = force.getSubset(region=refpt)
                                for i_deg in [1,2,3]:
                                    instances_dict[i_instance][step_name]["REF{}, RF{}".format(ref_pt, i_deg)] = float(ref_pt_rf.values[0].data[i_deg-1])
                                    instances_dict[i_instance][step_name]["REF{}, U{}".format(ref_pt, i_deg)] = float(ref_pt_u.values[0].data[i_deg-1])

                odict['Instances'] = instances_dict

                with open(FILEOUTPUT1,'w') as fout:
                    json.dump(odict, fout, indent=4)
                count += 1
                print >> log_file, 'i = {}'.format(str(count))
                odb.close()
                
        with open('odb_extraction_completed.log', 'w') as fout:
            fout.write('')
            
###########################################################################
data_extractor()

