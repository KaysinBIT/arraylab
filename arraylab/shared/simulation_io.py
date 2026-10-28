"""Bounded, pickle-free numerical artifacts shared by independent simulators."""
from pathlib import Path
import ast
import hashlib
import json
import math
import os
import tempfile
import struct
import zipfile

import numpy as np

DEFAULT_MAX_BYTES=1024*1024*1024
MAX_MEMBERS=256


def clean_json(value, *, complex_style='object'):
    if complex_style not in ('object','pair'):raise ValueError('Unknown complex representation')
    if isinstance(value,dict):return {str(k):clean_json(v,complex_style=complex_style) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(v,complex_style=complex_style) for v in value]
    if isinstance(value,np.ndarray):return clean_json(value.tolist(),complex_style=complex_style)
    if isinstance(value,np.generic):return clean_json(value.item(),complex_style=complex_style)
    if isinstance(value,complex):
        pair=[clean_json(value.real),clean_json(value.imag)]
        return {'real':pair[0],'imag':pair[1]} if complex_style=='object' else pair
    if isinstance(value,float) and not math.isfinite(value):return None
    if value is None or type(value) in (str,bool,int,float):return value
    if isinstance(value,Path):return str(value)
    raise TypeError('Unsupported result value: '+type(value).__name__)


def json_bytes(value,*,complex_style='object'):
    return (json.dumps(clean_json(value,complex_style=complex_style),ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf-8')


def _atomic(path,writer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:
            writer(stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def write_json(path,value,*,complex_style='object'):
    data=json_bytes(value,complex_style=complex_style)
    _atomic(path,lambda stream:stream.write(data))
    return Path(path)


def _parse_json(text):
    def finite_float(value):
        result=float(value)
        if not math.isfinite(result):raise ValueError('JSON数值必须有限：'+value)
        return result
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('JSON存在重复字段：'+key)
            result[key]=value
        return result
    try:return json.loads(text,object_pairs_hook=pairs,parse_float=finite_float,
                          parse_constant=lambda v:(_ for _ in ()).throw(ValueError('JSON数值必须有限：'+v)))
    except RecursionError as exc:raise ValueError('JSON嵌套过深。') from exc


def read_json(path,max_bytes=64*1024*1024):
    with Path(path).open('rb') as stream:data=stream.read(max_bytes+1)
    if len(data)>max_bytes:raise ValueError('JSON文件超过读取预算。')
    return _parse_json(data.decode('utf-8-sig'))


def write_npz(path,result,arrays,*,complex_style='object',compressed=True):
    if type(result) is not dict:raise ValueError('结果元数据须为对象。')
    if len(arrays)>=MAX_MEMBERS:raise ValueError('数组数量超出记录格式预算。')
    values={}
    for key,value in arrays.items():
        if not isinstance(key,str) or not key or key=='metadata_json' or '/' in key or '\\' in key or '\x00' in key:
            raise ValueError('数组名称无效。')
        array=np.asarray(value)
        if array.dtype.hasobject or array.ndim>8:raise ValueError('数组不得包含Python对象，且最多8维。')
        values[key]=array
    values['metadata_json']=np.array(json_bytes(result,complex_style=complex_style).decode('utf-8'))
    # NumPy's keyword writer reserves names such as file/allow_pickle. ZIP
    # members have no such restriction, so use the actual public array names.
    def writer(stream):
        compression=zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
        with zipfile.ZipFile(stream,'w',compression=compression,allowZip64=True) as archive:
            for name,array in values.items():
                with archive.open(name+'.npy','w',force_zip64=True) as member:
                    np.lib.format.write_array(member,array,allow_pickle=False)
    _atomic(path,writer)
    return Path(path)


def _npy_header(stream):
    """Validate a bounded header before NumPy is allowed to allocate arrays."""
    version=np.lib.format.read_magic(stream)
    if version==(1,0):return np.lib.format.read_array_header_1_0(stream,max_header_size=16384)
    if version==(2,0):return np.lib.format.read_array_header_2_0(stream,max_header_size=16384)
    if version!=(3,0):raise ValueError('不支持此NPY数组头版本。')
    # NPY 3 uses the v2 four-byte length but UTF-8 field names. NumPy has no
    # public v3 header-only reader; parse just this fixed, documented envelope.
    length=stream.read(4)
    if len(length)!=4:raise ValueError('NPY数组头不完整。')
    size=struct.unpack('<I',length)[0]
    if size>16384:raise ValueError('NPY数组头超过读取预算。')
    raw=stream.read(size)
    if len(raw)!=size:raise ValueError('NPY数组头不完整。')
    try:header=ast.literal_eval(raw.decode('utf-8'))
    except (ValueError,SyntaxError,UnicodeError,RecursionError) as exc:raise ValueError('NPY数组头格式无效。') from exc
    if (type(header) is not dict or set(header)!={'shape','fortran_order','descr'} or
            type(header['shape']) is not tuple or any(type(n) is not int or n<0 for n in header['shape']) or
            type(header['fortran_order']) is not bool):raise ValueError('NPY数组头字段无效。')
    return header['shape'],header['fortran_order'],np.lib.format.descr_to_dtype(header['descr'])


def read_npz(path,max_bytes=DEFAULT_MAX_BYTES):
    """Read a path or caller-owned seekable binary stream as one file snapshot."""
    with zipfile.ZipFile(path) as archive:
        items=archive.infolist()
        if (len(items)>MAX_MEMBERS or len({x.filename for x in items})!=len(items) or
                sum(x.file_size for x in items)>max_bytes or
                any(not x.filename.endswith('.npy') or len(x.filename)==4 or '/' in x.filename or '\\' in x.filename for x in items)):
            raise ValueError('数组文件结构或展开大小超出读取预算。')
        for item in items:
            with archive.open(item) as stream:
                shape,_,dtype=_npy_header(stream)
                if dtype.hasobject or len(shape)>8 or math.prod(shape)*dtype.itemsize!=item.file_size-stream.tell():
                    raise ValueError('数组头与实际数据大小不一致或包含对象。')
        members={item.filename[:-4]:item for item in items}
        if 'metadata_json' not in members:raise ValueError('数组文件缺少结果元数据。')
        def read(name):
            with archive.open(members[name]) as member:
                return np.lib.format.read_array(member,allow_pickle=False,max_header_size=16384)
        meta=read('metadata_json')
        if meta.ndim!=0 or meta.dtype.kind not in ('U','S'):raise ValueError('数组元数据格式无效。')
        text=meta.item();text=text.decode('utf-8') if isinstance(text,bytes) else text
        if len(text)>64*1024*1024:raise ValueError('数组元数据超过读取预算。')
        result=_parse_json(text)
        if type(result) is not dict:raise ValueError('结果元数据须为对象。')
        arrays={key:read(key) for key in members if key!='metadata_json'}
    if any(value.dtype.hasobject for value in arrays.values()):raise ValueError('不支持对象数组。')
    return result,arrays


def configuration_digest(configuration):
    return hashlib.sha256(json.dumps(configuration,sort_keys=True,ensure_ascii=False,
                                   allow_nan=False,separators=(',',':')).encode('utf-8')).hexdigest()
