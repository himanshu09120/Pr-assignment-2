"""Bundle report, code, measured outputs, and unique original input resources."""
from pathlib import Path
import hashlib
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    paths=[]
    for name in ('README.md','requirements.txt','config.json','.gitignore'):
        paths.append(ROOT/name)
    paths.extend(ROOT.glob('*.pdf'))
    paths.append(ROOT/'Data/readme.txt')
    paths.extend((ROOT/'Data').glob('sub*/Normal/txt/*.txt'))
    paths.extend((ROOT/'Data').glob('sub*/Aggressive/txt/*.txt'))
    for directory in ('src','matlab','tests','report','artifacts'):
        for path in (ROOT/directory).rglob('*'):
            relative=path.relative_to(ROOT)
            if not path.is_file() or any(part in ('__pycache__','cache','reference_text') for part in relative.parts):
                continue
            if path.suffix in ('.pyc','.aux','.out') or path.name=='CHECKSUMS.sha256':
                continue
            if path.suffix=='.log' and relative.parts[0]!='artifacts':
                continue
            paths.append(path)
    paths=sorted(set(paths))
    checksum=ROOT/'artifacts/CHECKSUMS.sha256'
    checksum.write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))
                                  for p in paths)+'\n')
    paths.append(checksum)
    target=ROOT/'EMG_CA2_Submission.zip'
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in paths:
            archive.write(path,str(path.relative_to(ROOT)))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        names=archive.namelist()
        assert not any('sub3/sub3' in name for name in names)
        assert sum('/txt/' in name for name in names)==80
        assert 'report/EMG_Classification_Report.pdf' in names
    print(f'{target.name}: {len(paths)} files, {target.stat().st_size/1024**2:.2f} MiB; CRC check passed')


if __name__=='__main__':
    main()
