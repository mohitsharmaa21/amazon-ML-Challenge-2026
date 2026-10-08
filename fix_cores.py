with open('code/business_entity_resolution/src/er_entilink_pipeline.py', 'r', encoding='utf-8') as f: code = f.read()
code = code.replace('ap.add_argument("--min-name-len"', 'ap.add_argument("--num-cores", type=int, default=1, help="number of cores for entilink")\n    ap.add_argument("--min-name-len"')
code = code.replace('num_cores=1', 'num_cores=args.num_cores')
with open('code/business_entity_resolution/src/er_entilink_pipeline.py', 'w', encoding='utf-8') as f: f.write(code)
