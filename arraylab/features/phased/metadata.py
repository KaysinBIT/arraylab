"""Shared native-form metadata; units and angle conventions stay explicit."""
GROUPS={'array':'阵列与激励','scan':'扫描性能评估','linear':'一维对称位置综合','planar':'二维联合综合','mapSize':'显示采样'}
ACTIONS=[('analysis','更新阵列分析'),('scan','评估扫描性能'),('optimize1d','一维稀疏综合'),('optimize2d','二维联合综合')]
VIEWS=[('layout','阵列幅相布局'),('cuts','X/Y切面与指标'),('uv','二维u–v方向图'),('scan','扫描性能'),('optimization','优化过程'),('cases','逐工况约束'),('elements','逐阵元激励')]
CHOICES={'array.geometry':[('直线阵','linear'),('矩形平面阵','rectangular'),('自定义坐标','custom')],
         'array.window':[('均匀','uniform'),('Hann','hann'),('Hamming','hamming'),('Taylor','taylor')],
         'array.phaseMode':[('固定参考频率移相','phase'),('理想真时延','ttd')],
         'planar.mode':[('对称格点抽稀','grid'),('连续位置与幅度','continuous')],
         'planar.initialization':[('有界自动生成','generated'),('使用提供初值','provided')]}
LABELS={'array.name':'方案名称','array.geometry':'阵列几何','array.nx':'X阵元数','array.ny':'Y阵元数','array.dx':'X间距 / λ0','array.dy':'Y间距 / λ0',
        'array.referenceGHz':'几何/移相参考频率 / GHz','array.frequencyGHz':'分析频率 / GHz','array.steerX':'命令X方向角 / °','array.steerY':'命令Y方向角 / °',
        'array.window':'幅度窗','array.sidelobeDb':'Taylor设计旁瓣 / dB','array.phaseBits':'移相位数（0=连续）','array.phaseMode':'移相方式',
        'array.elementPower':'单元功率cos(theta)指数','array.custom':'自定义阵元（λ0、相对电压、度）',
        'scan.maxAngle':'X扫描中心半角 / °','scan.step':'扫描步长 / °','linear.count':'偶数阵元数','linear.aperture':'固定总孔径 / λ0',
        'linear.minSpacing':'最小间距 / λ0','linear.scanMax':'验证扫描半角 / °','linear.targetSll':'切面旁瓣上限 / dB','linear.maxHpbw':'HPBW上限 / °',
        'linear.iterations':'搜索迭代预算','linear.seed':'随机种子','planar.mode':'二维综合模式','planar.count':'偶数阵元数','planar.gridNx':'候选X格点数',
        'planar.gridNy':'候选Y格点数','planar.dx':'候选X格点间距 / λ0','planar.dy':'候选Y格点间距 / λ0','planar.minSpacing':'欧氏最小间距 / λ0',
        'planar.width':'连续X包络宽度 / λ0','planar.height':'连续Y包络宽度 / λ0','planar.amplitudeMin':'正电压幅度下限','planar.optimizePositions':'优化连续坐标',
        'planar.optimizeAmplitudes':'优化共享幅度','planar.initialization':'连续模式初值来源','planar.initialPositions':'反演对称初始阵元',
        'planar.scans':'成对X/Y扫描方向角 / °','planar.frequencies':'联合频率 / GHz','planar.targetSll':'二维掩膜外旁瓣上限 / dB',
        'planar.maxHpbwX':'X切面HPBW上限 / °','planar.maxHpbwY':'Y切面HPBW上限 / °','planar.mainlobeRadiusU':'参考u排除半径',
        'planar.mainlobeRadiusV':'参考v排除半径','planar.iterations':'迭代预算（0仅验证初值）','planar.seed':'随机种子','mapSize':'u–v显示网格边长',
        'x':'x / λ0','y':'y / λ0','amplitude':'相对电压幅度','phaseDeg':'附加相位 / °'}
HELP={'array':'坐标单位为参考波长λ0。X/Y为方向角：u=sin(X)、v=sin(Y)，两者平方和不得超过1。',
      'array.referenceGHz':'改变参考频率会改变归一化几何对应的物理尺寸；仅改变分析频率保持物理尺寸。',
      'array.phaseMode':'固定移相保留参考频率相位；TTD逐频理想校相。TTD后叠加相位量化不等于固定时延量化器。',
      'array.custom':'每个阵元含x,y,amplitude,phaseDeg。自定义阵列直接使用给定激励，不再叠加窗。',
      'scan':'固定命令Y方向余弦的X切面；不保证穿过真实二维峰。空值保留原因，不当作0。',
      'planar':'共享一组物理坐标与电压幅度，最多25个频率×方向工况；最终独立细验证，不承诺连续角域或全局最优。',
      'planar.initialPositions':'连续初值必须反演对称、同对等幅、真实间距达标、正幅度且最大值为1，不自动归一化输入。',
      'planar.mainlobeRadiusU':'区域旁瓣是在显式排除椭圆外评价，不是第一零点旁瓣；半径随f0/f缩放。'}
LIST_DEFAULTS={'array.custom':{'x':0.,'y':0.,'amplitude':1.,'phaseDeg':0.},'planar.initialPositions':{'x':.5,'y':.5,'amplitude':1.,'phaseDeg':0.},'planar.scans':{'x':0.,'y':0.},'planar.frequencies':10.}
