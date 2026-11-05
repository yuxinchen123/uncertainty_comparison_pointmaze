"""Prepare one direct-controller transport configuration from real reviewed unit metadata.

This writes only the exact new transport source copy and its configuration. It does
not create logs, change permissions, contact the target, or start any process.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re


D = Path(__file__).resolve().parent
specification = importlib.util.spec_from_file_location('profile_qualification_transport_preparation',
    D / 'start_direct_profile_qualification_once.py')
transport = importlib.util.module_from_spec(specification)
specification.loader.exec_module(transport)


def write_once(path, raw):
    """Keep every attempted actual configuration distinct and durable."""
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def prepare(packet, controller, procedure, visibility):
    """Derive all plan/source/standard references from the actual bound controller."""
    packet, controller = Path(packet).absolute(), Path(controller).absolute()
    run = packet.parent.parent
    transport.require(packet.parent == run/'remote_setup' and packet.resolve(strict=True) == packet
        and controller.parent == run/'data' and controller.resolve(strict=True) == controller,
        'the existing packet and controller must belong to the same exact campaign')
    def reference(path):return transport.capture(path)[1]
    parent_ref = reference(controller/'measurement_plan.json')
    parent = transport.document(parent_ref)
    binding_ref = reference(controller/'prepared_child_binding.json')
    binding = transport.document(binding_ref)
    child = transport.document(binding['child_plan'])
    review_ref = reference(controller/'controller_launch_independent_review.json')
    review = transport.document(review_ref)
    transport.require(parent['kind'] == 'profile_target_qualification_controller'
        and child['kind'] == 'full_original_profile_source_target_qualification'
        and child['hardware_class'] == 'h100' and child['copy_seeds'] == [1]
        and type(child['copy_seeds'][0]) is int and child['configuration']['learner']['n_copies'] == 1
        and binding['controller_plan'] == parent_ref and parent['profile_unit'] == str(Path(binding['child_plan']['path']).parent)
        and review.get('kind') == 'independent_profile_target_qualification_controller_launch_review'
        and review.get('approved') is review.get('passed') is True and review.get('independent_of_producer') is True
        and review.get('controller_plan') == parent_ref and review.get('child_plan') == binding['child_plan'],
        'only the actual reviewed one-copy H100 qualification controller can be bound')
    procedure_ref, visibility_ref = reference(procedure), reference(visibility)
    evidence = transport.document(visibility_ref)
    log = run/'slurm/direct_jobs'/packet.name
    transport.require(log.resolve(strict=True) == log and log.stat().st_uid == os.getuid()
        and evidence.get('path') == str(log) and evidence.get('hostname') == 'serval02'
        and evidence.get('target_visibility_verified') is True and evidence.get('owner_uid') == transport.UID,
        'root must first create the actual exact log directory and record its direct-host visibility')
    paths = (packet/'start_direct_profile_qualification_once.py', packet/'root_transport_configuration.json')
    transport.require(not any(os.path.lexists(path) for path in paths)
        and not any(os.path.lexists(packet/name) for name in transport.ATTEMPTS)
        and not any(os.path.lexists(log/name) for name in ('controller_stdout.raw','controller_stderr.raw')),
        'never regenerate an attempted source, configuration or log')
    standards = [reference(folder/name) for folder in (packet,controller)
        for name in ('manifest.yaml','config_resolved.yaml','command.txt','experiment_background.md')]
    raw = (D/'start_direct_profile_qualification_once.py').read_bytes()
    write_once(paths[0],raw)
    value = {'schema_version':1,'kind':'direct_profile_target_qualification_root_transport',
        'automatic_retry_permitted':False,'packet':str(packet),'campaign_run_dir':str(run),
        'controller_unit':str(controller),'transport_source':reference(paths[0]),'controller_plan':parent_ref,
        'child_plan':binding['child_plan'],'prepared_child_binding':binding_ref,'launch_review':review_ref,
        'controller_source':reference(controller/'code/profile_target_qualification.py'),
        'process_source':reference(controller/'code/direct_common_timing_controller.py'),
        'metadata_permission_procedure':procedure_ref,'log_directory_evidence':visibility_ref,
        'standard_records':standards,'stdout':str(log/'controller_stdout.raw'),'stderr':str(log/'controller_stderr.raw')}
    write_once(paths[1],(json.dumps(value,indent=2,sort_keys=True)+'\n').encode())
    return {'configuration':reference(paths[1]),'source':reference(paths[0]),'actual_transport_attempted':False,
        'required_next_review_kind':'independent_direct_profile_target_qualification_root_transport_review',
        'publication_required_before_actual_launch':True,'standing_gpu_descriptor_opened':False}


def main(argv=None):
    """Require the current exact owned CPU job and an explicit write request."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('packet','controller-unit','permission-procedure','log-directory-evidence'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--owned-job-id',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'action_taken':False,'actual_transport_attempted':False}));return
    job,step=args.owned_job_id,os.environ.get('SLURM_STEP_ID','')
    transport.require(re.fullmatch('[1-9][0-9]*',job) is not None and os.environ.get('SLURM_JOB_ID') == job
        and re.fullmatch('[0-9]+',step) is not None
        and '/job_'+job+'/step_'+step+'/' in (Path('/proc/self/cgroup').read_text()+'/'),
        'transport metadata preparation requires the declared exact owned CPU step')
    run=args.packet.parent.parent
    transport.require((run/'slurm/submitted_jobids_uva.txt').read_text().splitlines().count(job)==1,
        'the current CPU job must occur once in this campaign owned ledger')
    print(json.dumps(prepare(args.packet,args.controller_unit,args.permission_procedure,args.log_directory_evidence)))


if __name__=='__main__':main()
