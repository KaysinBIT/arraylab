"""Shared simulator artifacts: exact arrays, bounded input and atomic writes."""
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from arraylab.shared import simulation_io as io


@pytest.mark.parametrize('style,expected',[('pair',[1.,2.]),('object',{'real':1.,'imag':2.})])
def test_complex_styles_numpy_scalars_and_unknown_values(style,expected):
    value={'scalar':np.complex128(1+2j),'values':np.array([np.nan,np.inf,-np.inf]),'integer':np.int64(4)}
    actual=io.clean_json(value,complex_style=style)
    assert actual=={'scalar':expected,'values':[None,None,None],'integer':4}
    assert json.loads(io.json_bytes(value,complex_style=style))==actual
    assert value['scalar']==1+2j and np.isnan(value['values'][0])


@pytest.mark.parametrize('style',['pair','object'])
@pytest.mark.parametrize('compressed',[False,True])
def test_npz_roundtrip_keeps_dtype_shape_and_complex_signal(tmp_path,style,compressed):
    arrays={'raw':np.arange(24,dtype=np.float64).reshape(2,3,4)*(1+2j),
            'mask':np.array([[True,False]]),'scalar':np.array(3.),'labels':np.array(['λ','目标'])}
    before={name:value.copy() for name,value in arrays.items()}
    result={'format':'record','phase':1+2j,'missing':np.nan,'nested':{'gain':np.float64(3.5)}}
    target=tmp_path/'完整 记录.npz';io.write_npz(target,result,arrays,complex_style=style,compressed=compressed)
    saved,loaded=io.read_npz(target)
    assert saved==io.clean_json(result,complex_style=style) and set(loaded)==set(arrays)
    for name in arrays:
        assert loaded[name].dtype==arrays[name].dtype and loaded[name].shape==arrays[name].shape
        np.testing.assert_array_equal(loaded[name],arrays[name]);np.testing.assert_array_equal(arrays[name],before[name])


@pytest.mark.parametrize('name',['','metadata_json','../outside','dir/data','dir\\data',3])
def test_invalid_array_names_are_rejected_without_files(tmp_path,name):
    target=tmp_path/'invalid.npz'
    with pytest.raises(ValueError):io.write_npz(target,{}, {name:np.ones(2)})
    assert not target.exists() and not list(tmp_path.iterdir())


def test_object_arrays_never_enter_npz(tmp_path):
    target=tmp_path/'objects.npz'
    with pytest.raises(ValueError,match='对象'):io.write_npz(target,{}, {'raw':np.array([{'bad':1}],dtype=object)})
    assert not target.exists()


def test_atomic_json_and_npz_failure_preserve_existing_files(tmp_path,monkeypatch):
    target=tmp_path/'record.npz';target.write_bytes(b'previous-valid-record')
    before=target.read_bytes()
    original_fsync=io.os.fsync
    def interrupted(fd):raise OSError('simulated disk synchronization failure')
    monkeypatch.setattr(io.os,'fsync',interrupted)
    with pytest.raises(OSError):io.write_npz(target,{}, {'value':np.arange(5)})
    assert target.read_bytes()==before and list(tmp_path.iterdir())==[target]
    monkeypatch.setattr(io.os,'fsync',original_fsync)
    text=tmp_path/'config.json';text.write_bytes(b'previous-config')
    def replacement(*args):raise OSError('simulated replace failure')
    monkeypatch.setattr(io.os,'replace',replacement)
    with pytest.raises(OSError):io.write_json(text,{'new':1})
    assert text.read_bytes()==b'previous-config'
    assert {p.name for p in tmp_path.iterdir()}=={'record.npz','config.json'}


def npy_bytes(shape,dtype,payload=b''):
    stream=BytesIO()
    np.lib.format.write_array_header_1_0(stream,{'descr':np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                              'fortran_order':False,'shape':shape})
    stream.write(payload);return stream.getvalue()


def metadata_bytes(text='{}'):
    stream=BytesIO();np.save(stream,np.array(text),allow_pickle=False);return stream.getvalue()


@pytest.mark.parametrize('shape,dtype,payload',[((10**12,),'<f8',b''),((1,),object,b'notpickle'),
                                               ((0,)*9,'<f8',b''),((2,),'<f8',b'\0'*8)])
def test_untrusted_shape_object_or_truncated_payload_rejected_before_numpy_load(tmp_path,monkeypatch,shape,dtype,payload):
    target=tmp_path/'malformed.npz'
    with zipfile.ZipFile(target,'w') as archive:
        archive.writestr('raw.npy',npy_bytes(shape,dtype,payload));archive.writestr('metadata_json.npy',metadata_bytes())
    monkeypatch.setattr(io.np,'load',lambda *a,**k:pytest.fail('Untrusted header must be rejected before loading'))
    with pytest.raises(ValueError):io.read_npz(target)


@pytest.mark.parametrize('name',['../raw.npy','raw.txt','dir/raw.npy','dir\\raw.npy'])
def test_archive_member_paths_and_types_rejected(tmp_path,name):
    target=tmp_path/'malformed.npz'
    with zipfile.ZipFile(target,'w') as archive:
        archive.writestr(name,npy_bytes((1,),'<f8',b'\0'*8));archive.writestr('metadata_json.npy',metadata_bytes())
    with pytest.raises(ValueError):io.read_npz(target)


def test_zip_duplicate_members_and_uncompressed_budget_are_checked(tmp_path):
    target=tmp_path/'duplicate.npz'
    with zipfile.ZipFile(target,'w') as archive:
        archive.writestr('metadata_json.npy',metadata_bytes())
        with pytest.warns(UserWarning,match='Duplicate'):archive.writestr('metadata_json.npy',metadata_bytes())
    with pytest.raises(ValueError):io.read_npz(target)
    io.write_npz(target,{}, {'zeros':np.zeros(10000)})
    assert target.stat().st_size<10000
    with pytest.raises(ValueError,match='大小'):io.read_npz(target,max_bytes=2000)


@pytest.mark.parametrize('text',['{"x":NaN}','{"x":Infinity}','{"x":1,"x":2}'])
def test_json_and_npz_metadata_use_same_strict_json_contract(tmp_path,text):
    document=tmp_path/'invalid.json';document.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError):io.read_json(document)
    target=tmp_path/'invalid.npz'
    with zipfile.ZipFile(target,'w') as archive:archive.writestr('metadata_json.npy',metadata_bytes(text))
    with pytest.raises(ValueError):io.read_npz(target)


@pytest.mark.parametrize('text',['[]','null','"not a result"'])
def test_npz_result_metadata_must_be_an_object(tmp_path,text):
    target=tmp_path/'invalid.npz'
    with zipfile.ZipFile(target,'w') as archive:archive.writestr('metadata_json.npy',metadata_bytes(text))
    with pytest.raises(ValueError):io.read_npz(target)


def test_json_byte_limit_and_bom_support(tmp_path):
    target=tmp_path/'bom.json';target.write_bytes(b'\xef\xbb\xbf'+b'{"ok": true}')
    assert io.read_json(target)=={'ok':True}
    with pytest.raises(ValueError,match='预算'):io.read_json(target,max_bytes=3)


def test_npz_arrays_survive_closed_archive_without_extra_copy(tmp_path):
    target=tmp_path/'closed.npz';source=np.arange(12,dtype=np.complex128).reshape(3,4)
    io.write_npz(target,{'kind':'owned'}, {'raw':source})
    result,arrays=io.read_npz(target)
    target.unlink()  # Proves all archive handles are released on Windows too.
    np.testing.assert_array_equal(arrays['raw'],source)
    arrays['raw'][0,0]=99j
    assert source[0,0]==0 and result=={'kind':'owned'}
