"""Two critical paths: actual parameter updates and exact paired disk resume."""
import copy
import json
import traceback
import numpy as np
import torch
import continue_train as v
import micro


class FixtureDataset:
    def bridge(self,parent):return micro.fixture(300003,'near')[0]


def main():
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    output=v.HERE/'validation.json';folder=v.HERE/'validation_artifacts'
    if output.exists() or folder.exists():raise FileExistsError('Validation already exists')
    folder.mkdir();result=dict(status='RUNNING',device='cuda:0',scope='24 disposable fixture joint steps; not formal training',arms={})
    try:
        ds=FixtureDataset();parent_states=[]
        for arm in v.CONFIG['arms']:
            run=v.PairedRun(ds,77126,arm,'cuda:0')
            parent_states.append(run.run.snapshot())
            run.run.bridge=ds.bridge(run.run.parent)
            data,trace=run.collect(4)
            stats=run.update(data,1)
            changed={g:stats['actor_before'][g]!=stats['actor_after'][g] for g in v.GROUPS}
            if arm=='joint':
                assert all(changed.values()),changed
                assert all(n>0 for n in stats['gradient_norms'].values())
            else:
                assert changed['rescue'] and not any(changed[g] for g in v.GROUPS if g!='rescue')
            saved=run.snapshot({});cp=folder/f'{arm}_resume.pt';v.s.checkpoint_write(cp,saved)
            next_data,next_trace=run.collect(4);run.update(next_data,1);expected=run.snapshot({})
            other=v.PairedRun(ds,77126,arm,'cuda:0')
            other.restore(torch.load(cp,map_location='cuda:0',weights_only=False))
            actual_data,actual_trace=other.collect(4);other.update(actual_data,1);actual=other.snapshot({})
            assert v.s.same(next_trace,actual_trace)
            assert all(np.array_equal(a,b) for a,b in zip(next_data,actual_data))
            assert v.s.same(expected,actual),'Resumed state differs'
            result['arms'][arm]=dict(parameter_groups_changed=changed,gradient_norms=stats['gradient_norms'],
                disk_resume_model_optimizer_valuenorm_environment_rng_exact=True)
        assert v.s.same(parent_states[0],parent_states[1]),'Paired arms must inherit identical full Run states'
        result.update(status='PASS',paired_initial_state_exact=True)
    except BaseException:
        result.update(status='FAIL',error=traceback.format_exc());v.s.json_write(output,result);raise
    v.s.json_write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
