from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader,KlineLoader
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'kline': KlineLoader,
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    
    # 使用 getattr 提供默认值
    percent = getattr(args, 'percent', 100)  # 默认使用100%的数据
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=args.freq,
        percent=percent,
        seasonal_patterns=args.seasonal_patterns,
        args=args
    )
    print(flag, len(data_set))
    
    # 准备 DataLoader 参数
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle_flag,
        'num_workers': args.num_workers,
        'drop_last': drop_last
    }
    
    # 只有 UEA 数据需要特殊的 collate_fn
    if args.data == 'UEA':
        from data_provider.uea import collate_fn
        loader_kwargs['collate_fn'] = lambda x: collate_fn(x, max_len=args.seq_len)
    
    data_loader = DataLoader(data_set, **loader_kwargs)
    
    return data_set, data_loader