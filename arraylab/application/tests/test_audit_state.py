from pathlib import Path
from io import BytesIO
import os,zipfile
import struct
import numpy as np
import pytest
from arraylab.shared import simulation_io as io
from arraylab.application import simulation

@pytest.mark.parametrize('number',['1e999','-1e999'])
@pytest.mark.parametrize('kind',['json','npz'])
def test_rejects_overflow_json_numbers(tmp_path,number,kind):
 text='{"value":'+number+'}'
 path=tmp_path/('invalid.'+kind)
 if kind=='json':path.write_text(text,encoding='utf-8');read=io.read_json
 else:
  b=BytesIO();np.save(b,np.array(text),allow_pickle=False)
  with zipfile.ZipFile(path,'w') as archive:archive.writestr('metadata_json.npy',b.getvalue())
  read=io.read_npz
 with pytest.raises(ValueError,match='有限'):read(path)

@pytest.mark.parametrize('compressed',[False,True])
def test_exact_member_names_do_not_alias(tmp_path,compressed):
 arrays={'raw':np.array([1.,2.]),'raw.npy':np.array([8.,9.]),'metadata_json.npy':np.array([20.,30.])}
 path=tmp_path/'names.npz';io.write_npz(path,{'kind':'names'},arrays,compressed=compressed)
 result,loaded=io.read_npz(path)
 assert result=={'kind':'names'} and set(loaded)==set(arrays)
 for key,value in arrays.items():np.testing.assert_array_equal(loaded[key],value)

@pytest.mark.parametrize('name',['allow_pickle','file'])
def test_numpy_writer_options_are_legitimate_array_names(tmp_path,name):
 path=tmp_path/'names.npz';array=np.arange(4)
 io.write_npz(path,{}, {name:array});_,loaded=io.read_npz(path)
 assert set(loaded)=={name};np.testing.assert_array_equal(loaded[name],array)


def test_sidecar_digest_and_arrays_use_same_file_generation(tmp_path,monkeypatch):
 config=simulation.provider('phased').default_configuration();config['array'].update(geometry='linear',nx=2,ny=1);config['mapSize']=21
 record,arrays=simulation.calculate('phased','analysis',config,workers=1)
 target=tmp_path/'saved';simulation.export_result(target,record,arrays)
 changed={k:v.copy() for k,v in arrays.items()};changed['cut_x'][0,1]=123.
 replacement=tmp_path/'replacement.npz';io.write_npz(replacement,record['result'],changed)
 reader=simulation.read_npz
 def racing_read(path,*args,**kwargs):
  try:os.replace(replacement,target/'arrays.npz')
  except PermissionError:pass # Windows denies replacement while the same opened generation is held.
  return reader(path,*args,**kwargs)
 monkeypatch.setattr(simulation,'read_npz',racing_read)
 restored,actual=simulation.load_record(target/'arraylab-record.json')
 assert restored==record
 for name,array in arrays.items():np.testing.assert_array_equal(actual[name],array)


@pytest.mark.parametrize('compressed',[False,True])
def test_utf8_structured_array_and_fortran_order_roundtrip(tmp_path,compressed):
 arrays={'sensors':np.array([(1.5,3)],dtype=[('温度','f8'),('计数','i4')]),
         'matrix':np.asfortranarray(np.arange(12.).reshape(3,4))}
 path=tmp_path/'structured.npz'
 with pytest.warns(UserWarning,match='format 3.0'):io.write_npz(path,{'name':'UTF-8'},arrays,compressed=compressed)
 result,loaded=io.read_npz(path)
 assert result=={'name':'UTF-8'} and loaded['matrix'].flags.f_contiguous
 for key,value in arrays.items():
  assert loaded[key].dtype==value.dtype;np.testing.assert_array_equal(loaded[key],value)


def test_npz_reader_keeps_caller_owned_stream_open(tmp_path):
 path=tmp_path/'stream.npz';io.write_npz(path,{'kind':'stream'},{'raw':np.arange(4)})
 with path.open('rb') as stream:
  result,arrays=io.read_npz(stream)
  assert not stream.closed and result['kind']=='stream'
 path.unlink();np.testing.assert_array_equal(arrays['raw'],np.arange(4))


def test_member_limit_is_shared_by_writer_and_reader(tmp_path):
 path=tmp_path/'limit.npz';arrays={str(i):np.empty(0) for i in range(255)}
 io.write_npz(path,{},arrays);assert len(io.read_npz(path)[1])==255
 old=path.read_bytes();arrays['one_too_many']=np.empty(0)
 with pytest.raises(ValueError,match='数量'):io.write_npz(path,{},arrays)
 assert path.read_bytes()==old


def test_v3_header_size_is_checked_before_allocating(tmp_path,monkeypatch):
 header=repr({'descr':[('温度','<f8')],'fortran_order':False,'shape':(10**12,)}).encode('utf-8')
 header+=b' '*((64-(12+len(header)+1)%64)%64)+b'\n'
 raw=np.lib.format.magic(3,0)+struct.pack('<I',len(header))+header
 metadata=BytesIO();np.save(metadata,np.array('{}'),allow_pickle=False)
 path=tmp_path/'huge.npz'
 with zipfile.ZipFile(path,'w') as archive:
  archive.writestr('raw.npy',raw);archive.writestr('metadata_json.npy',metadata.getvalue())
 monkeypatch.setattr(np.lib.format,'read_array',lambda *a,**k:pytest.fail('Header must be checked first'))
 with pytest.raises(ValueError,match='数组头'):io.read_npz(path)


def test_representable_finite_json_boundary_remains_valid(tmp_path):
 path=tmp_path/'finite.json';path.write_text('{"high":1.7976931348623157e308,"low":5e-324}',encoding='utf-8')
 assert io.read_json(path)=={'high':np.finfo(float).max,'low':np.nextafter(0.,1.)}


def test_nul_array_name_is_rejected_without_replacing_existing_record(tmp_path):
 path=tmp_path/'valid.npz';io.write_npz(path,{'kind':'previous'},{'raw':np.arange(3)})
 before=path.read_bytes()
 with pytest.raises(ValueError,match='名称'):io.write_npz(path,{'kind':'new'},{'a\x00b':np.arange(3)})
 assert path.read_bytes()==before and list(tmp_path.iterdir())==[path]
 result,arrays=io.read_npz(path);assert result=={'kind':'previous'}
 np.testing.assert_array_equal(arrays['raw'],np.arange(3))
