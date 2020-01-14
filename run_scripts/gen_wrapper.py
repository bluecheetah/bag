from typing import Dict, Any, List, cast

import sys
import argparse
from pathlib import Path

from pybag.enum import DesignOutput
from bag.io import read_yaml
from bag.core import BagProject
from bag.design.database import ModuleDB


def _info(etype, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(etype, value, tb)
    else:
        import pdb
        import traceback
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(etype, value, tb)
        print()
        # ...then start the debugger in post-mortem mode.
        pdb.post_mortem(tb)


sys.excepthook = _info


def parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate cell from spec file.')
    parser.add_argument('specs', help='YAML specs file name.')
    parser.add_argument('--no-sch', dest='gen_sch', action='store_false', default=True,
                        help='disable schematic, only netlist')
    parser.add_argument('--no-layout', dest='gen_lay', action='store_false', default=True,
                        help='disable layout.')
    parser.add_argument('-x', '--rcx', dest='run_rcx', action='store_true', default=False,
                        help='run RCX.')
    args = parser.parse_args()
    return args


def generate_wrapper(sch_db: ModuleDB, wrapper_params: Dict[str, Any],
                     cv_info_list: List, dut_netlist: str, gen_sch: bool) -> None:
    wrapper_lib = wrapper_params.pop('wrapper_lib')
    wrapper_cell = wrapper_params.pop('wrapper_cell')
    wrapper_impl_cell = wrapper_params.pop('impl_cell')
    wrapper_netlist_path = Path(wrapper_params.pop('netlist_file'))
    wrapper_netlist_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper_cls = sch_db.get_schematic_class(wrapper_lib, wrapper_cell)
    wrapper_master = sch_db.new_master(wrapper_cls, params=wrapper_params['params'])
    wrapper_list = [(wrapper_master, wrapper_impl_cell)]

    sch_db.batch_schematic(wrapper_list, output=DesignOutput.SPECTRE,
                           fname=str(wrapper_netlist_path), cv_info_list=cv_info_list,
                           cv_netlist=dut_netlist)
    print(f'wrapper_netlist: {str(wrapper_netlist_path)}')
    if gen_sch:
        sch_db.batch_schematic(wrapper_list, output=DesignOutput.SCHEMATIC)


def run_main(prj: BagProject, args: argparse.Namespace) -> None:
    specs = read_yaml(args.specs)

    lay_db = prj.make_template_db(specs['impl_lib']) if args.gen_lay else None
    sch_db = prj.make_module_db(specs['impl_lib'])
    cv_info = []
    dut_params = specs['dut_params']
    dut_netlist = prj.generate_cell(dut_params, lay_db=lay_db, sch_db=sch_db,
                                    gen_lay=args.gen_lay, gen_sch=args.gen_sch,
                                    cv_info_out=cv_info,
                                    run_rcx=args.run_rcx)

    print(f'dut_netlist: {dut_netlist}')
    wrapper_params = specs['wrapper_params']
    prj.replace_dut_in_wrapper(wrapper_params['params'], dut_params['impl_lib'],
                               dut_params['impl_cell'])
    generate_wrapper(sch_db, wrapper_params, cv_info, dut_netlist,
                     gen_sch=args.gen_sch)


if __name__ == '__main__':
    _args = parse_options()

    local_dict = locals()
    if 'bprj' not in local_dict:
        print('creating BAG project')
        _prj = BagProject()
    else:
        print('loading BAG project')
        _prj = local_dict['bprj']

    run_main(_prj, _args)
