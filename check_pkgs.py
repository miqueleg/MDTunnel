import importlib
mods=["openmmforcefields","openmm","openff.toolkit","rdkit","numpy","pandas"]
for m in mods:
    try:
        spec = importlib.util.find_spec(m)
        print(m, 'ok' if spec else 'missing')
    except Exception as e:
        print(m, 'missing')
