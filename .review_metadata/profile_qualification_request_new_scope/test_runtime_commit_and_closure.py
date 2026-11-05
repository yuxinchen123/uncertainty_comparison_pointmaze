"""Check only request-source and helper boundaries using explicit synthetic metadata premises."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

W=Path(__file__).resolve().parents[2]
BUILDERS=W/'.review_metadata/profile_target_operational_preparation'
BASE='62a597d509355a0f159843cf8601d1e5fda9a7ba'


def load(copies):
    """Load only the selected metadata request builder, with no preparation or runtime child."""
    name='derive_one_copy_qualification_request.py' if copies==1 else 'derive_two_copy_qualification_request.py'
    spec=importlib.util.spec_from_file_location('request_scope_'+str(copies),BUILDERS/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class FourContextBoundary(Exception):
    """Stop at the unchanged four-context validator, before any request output exists."""


class RuntimeSourceAndClosureTests(unittest.TestCase):
    """Synthetic upstream receipts test only the new request boundary and grant no authority."""
    def premises(self,temporary,module,copies,hardware):
        """Build visibly synthetic source receipts and stop before the real science validators."""
        directory=Path(temporary);run=directory/'campaign';(run/'remote_setup').mkdir(parents=True)
        helper_root=directory/'helper';script=helper_root/module.PREFIX/'profile_selected.py'
        script.parent.mkdir(parents=True);script.write_text('# Synthetic metadata-only helper premise.\n')
        unit='metadata_only_'+hardware+'_copies_'+str(copies)
        args=SimpleNamespace(helper_root=helper_root,run_dir=run,packet=run/'remote_setup'/unit,
            hardware_class=hardware,profile_unit_id=unit,controller_unit_id=unit+'_controller',
            common_proof=directory/'synthetic_common_proof.json',common_review=directory/'synthetic_common_review.json')
        configured=hardware=='jaguar03_rtx_a4500';count=36 if configured else 45
        names=[module.PREFIX+'profile_target_tool_scope.py']+[module.PREFIX+'synthetic_helper_'+str(index)+'.py' for index in range(count-1)]
        helper={'source_commit':BASE,'source_root':str(helper_root.resolve()),
            'files':{name:hashlib.sha256(name.encode()).hexdigest() for name in names}}
        def reference(path):
            return {'path':str(path),'sha256':hashlib.sha256(str(path).encode()).hexdigest()}
        review_ref=reference(directory/'synthetic_context_review.json');plan_ref=reference(directory/'synthetic_common_plan.json')
        accepted={'plan':plan_ref,'result':reference(directory/'synthetic_common_result.json')}
        common={'campaign_run_dir':str(run),'hardware_class':hardware,'copy_seeds':list(range(1,copies+1)),
            'configuration':{'learner':{'n_copies':copies}},'inputs':{}}
        proof={'contexts':[{'hardware_class':hardware,'copies':copies,'review':review_ref}]}
        values={str(args.common_proof):proof,review_ref['path']:accepted,plan_ref['path']:common}
        calls=[];helper_calls=[]
        def receipt(**keywords):
            helper_calls.append(keywords);return copy.deepcopy(helper)
        def adjacent(plan,name):
            self.assertEqual(name,'profile_selected');return SimpleNamespace(helper_receipt=receipt)
        def require_four_contexts(plan,proof_ref,review_ref):
            calls.append((plan,proof_ref,review_ref));raise FourContextBoundary('no request output is allowed')
        q=SimpleNamespace(reference=reference,checked=lambda ref:copy.deepcopy(values[ref['path']]),
            adjacent=adjacent,require_four_contexts=require_four_contexts)
        return args,helper,q,calls,helper_calls

    def test_missing_real_runtime_pin_refuses_before_git_or_path_reads(self):
        """An unknown commit cannot fall through to the old runtime or any source loader."""
        for copies in (1,2):
            module=load(copies)
            with self.subTest(copies=copies),patch.object(module,'RUNTIME',None),patch.object(module.subprocess,'check_output') as git:
                with self.assertRaisesRegex(ValueError,'has not been pinned'):module.runtime(Path('/not/an/actual/helper'))
                git.assert_not_called()

    def test_all_four_request_entries_reach_unchanged_context_gate(self):
        """Exact new source/root/45-or-36 receipt reaches the existing gate without writing a packet."""
        for copies in (1,2):
            for hardware in ('h100','jaguar03_rtx_a4500'):
                with self.subTest(copies=copies,hardware=hardware),tempfile.TemporaryDirectory() as temporary:
                    module=load(copies);args,helper,q,calls,helper_calls=self.premises(temporary,module,copies,hardware)
                    with patch.object(module,'RUNTIME',BASE),patch.object(module,'runtime',return_value=q):
                        with self.assertRaises(FourContextBoundary):module.derive(args)
                    self.assertEqual(len(calls),1);self.assertEqual(calls[0][0]['helpers'],helper)
                    configured=hardware=='jaguar03_rtx_a4500'
                    self.assertEqual(helper_calls,[{'configured':configured,'direct':not configured,'owned_processes':configured,
                        'direct_profile':not configured,'configured_native':configured,'configured_prerequisites':configured,'profile_target':True}])
                    self.assertFalse(args.packet.exists());self.assertFalse((args.run_dir/'data').exists())

    def test_old_or_incomplete_helper_receipts_refuse_before_context_gate(self):
        """Old counts, absent new scope and wrong immutable source/root refuse without any packet write."""
        for copies in (1,2):
            for hardware in ('h100','jaguar03_rtx_a4500'):
                for mutation in ('old_count','missing_scope','wrong_commit','wrong_root'):
                    with self.subTest(copies=copies,hardware=hardware,mutation=mutation),tempfile.TemporaryDirectory() as temporary:
                        module=load(copies);args,helper,q,calls,unused=self.premises(temporary,module,copies,hardware)
                        if mutation=='old_count':helper['files'].pop(next(name for name in helper['files'] if 'synthetic_helper_' in name))
                        elif mutation=='missing_scope':helper['files'][module.PREFIX+'synthetic_replacement.py']=helper['files'].pop(module.PREFIX+'profile_target_tool_scope.py')
                        elif mutation=='wrong_commit':helper['source_commit']=module.SOURCE
                        else:helper['source_root']=str(Path(temporary)/'different_helper')
                        with patch.object(module,'RUNTIME',BASE),patch.object(module,'runtime',return_value=q):
                            with self.assertRaisesRegex(ValueError,'complete target closure differs'):module.derive(args)
                        self.assertEqual(calls,[]);self.assertFalse(args.packet.exists())

    def test_every_existing_science_and_request_statement_remains_exact(self):
        """Only the runtime pin and two explicit guards differ from the actual published builders."""
        for copies in (1,2):
            module=load(copies);path=Path(module.__file__)
            before=ast.parse(subprocess.check_output(['git','-C',str(W),'show',BASE+':'+str(path.relative_to(W))],text=True))
            after=ast.parse(path.read_text())
            old_runtime=next(node for node in before.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='RUNTIME')
            for node in after.body:
                if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='RUNTIME':node.value=old_runtime.value
                if isinstance(node,ast.FunctionDef) and node.name in ('runtime','derive'):
                    node.body=[statement for statement in node.body if not (isinstance(statement,ast.Expr)
                        and isinstance(statement.value,ast.Call) and isinstance(statement.value.func,ast.Name)
                        and statement.value.func.id=='require' and any(isinstance(argument,ast.Constant)
                            and argument.value in ('the actual reviewed profile scope runtime commit has not been pinned',
                                'the exact reviewed profile scope helper source or complete target closure differs')
                            for argument in statement.value.args))]
            self.assertEqual(ast.dump(after,include_attributes=False),ast.dump(before,include_attributes=False))

    def test_transport_changes_only_runtime_pin_and_h100_closure_count(self):
        """Preserve every publication, ownership, process, permission and custody statement exactly."""
        path=BUILDERS/'start_direct_profile_qualification_once.py'
        before=ast.parse(subprocess.check_output(['git','-C',str(W),'show',BASE+':'+str(path.relative_to(W))],text=True))
        after=ast.parse(path.read_text())
        old_runtime=next(node for node in before.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='RUNTIME')
        for node in after.body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='RUNTIME':node.value=old_runtime.value
            if isinstance(node,ast.FunctionDef) and node.name=='verify':
                for statement in node.body:
                    if isinstance(statement,ast.Expr) and isinstance(statement.value,ast.Call) and any(isinstance(argument,ast.Constant)
                        and argument.value=='the exact reviewed H100 runtime helper closure is required' for argument in statement.value.args):
                        comparisons=[item for item in ast.walk(statement) if isinstance(item,ast.Compare)
                            and isinstance(item.left,ast.Call) and isinstance(item.left.func,ast.Name) and item.left.func.id=='len']
                        self.assertEqual(len(comparisons),1);self.assertEqual(comparisons[0].comparators[0].value,45)
                        comparisons[0].comparators[0].value=44
        self.assertEqual(ast.dump(after,include_attributes=False),ast.dump(before,include_attributes=False))


if __name__=='__main__':unittest.main(verbosity=2)
