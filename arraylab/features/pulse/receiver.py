"""Finite-record receiver operator and its exact noise adjoint.

The frontend is gated during transmit/recovery and starts each PRI with zero
filter state. ADC start is a recording crop, not a second RF switch: at most
len(b)-1 preceding samples are retained to evaluate its causal FIR history.
"""
from __future__ import annotations
import math
import numpy as np
from scipy import signal


def complete_delay_bounds(project,plan,taps):
    """Conservative full-envelope-and-FIR-tail reception delay interval (s)."""
    w,t=project['waveform'],project['timing']
    opening=max(float(t['receive_start_s']),float(w['pulse_width_s'])+float(t['recovery_s']))
    closing=min(float(t['receive_end_s']),float(np.min(plan['intervals_s'])))
    latest=closing-float(w['pulse_width_s'])-(len(taps)-1)/float(w['sample_rate_hz'])
    return opening,latest


def capture_layout(project, plan, taps):
    fs=float(project['waveform']['sample_rate_hz'])
    timing=project['timing'];start=float(timing['receive_start_s'])
    ends=np.minimum(np.asarray(plan['intervals_s'],float),float(timing['receive_end_s']))
    count=int(math.ceil((float(ends.max())-start)*fs-1e-10))
    if count<1:raise ValueError('接收窗没有有效采样')
    history=max(0,len(taps)-1)
    relative=start+(np.arange(count+history)-history)/fs
    gate=float(project['waveform']['pulse_width_s'])+float(timing['recovery_s'])
    input_valid=(relative[None,:]>=gate-1e-8/fs)&(relative[None,:]<ends[:,None]-1e-8/fs)
    output_valid=input_valid[:,history:]
    return {'relative_s':relative,'input_valid':input_valid,'output_valid':output_valid,
            'history_samples':history,'sample_count':count,'capture_start_s':start-history/fs}


def compression_origin_s(project,filter_length):
    """Physical delay at the first retained correlation lag, including negatives."""
    start=float(project['timing']['receive_start_s'])
    fs=float(project['waveform']['sample_rate_hz'])
    pre=0 if project['waveform']['kind']=='stepped' else min(filter_length-1,int(math.floor(start*fs+1e-10)))
    return start-pre/fs


def filter_capture(samples, taps, layout):
    values=np.asarray(samples,complex)
    gate=layout['input_valid']
    if values.ndim==3:gate=gate[:,None,:]
    filtered=signal.lfilter(taps,[1.],values*gate,axis=-1)
    output=filtered[...,layout['history_samples']:]
    final=layout['output_valid']
    if values.ndim==3:final=final[:,None,:]
    return output*final


def capture_adjoint(weights,taps,layout):
    """Return A^H w where raw=A n and n is independent input noise."""
    weights=np.asarray(weights,complex)
    final=layout['output_valid'];gate=layout['input_valid']
    if weights.ndim==3:final=final[:,None,:];gate=gate[:,None,:]
    padding=[(0,0)]*weights.ndim;padding[-1]=(layout['history_samples'],0)
    embedded=np.pad(weights*final,padding)
    backwards=signal.lfilter(np.conj(taps),[1.],embedded[...,::-1],axis=-1)[...,::-1]
    return backwards*gate


def receiver_filter(project):
    fs = float(project["waveform"]["sample_rate_hz"])
    bandwidth = float(project["radar"].get("receiver_bandwidth_hz",fs))
    if bandwidth>=fs*(1-1e-8): return np.ones(1)
    # Two-sided complex baseband passband. Symmetric causal FIR delay cancels
    # in correlation with the identically filtered waveform reference.
    return signal.firwin(33, min(.999,bandwidth/fs),window=("kaiser",7.0))
