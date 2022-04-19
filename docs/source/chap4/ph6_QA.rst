「6 学習を実行します」でエラーが発生する
=============================================
ファインチューニング用のモデルを読み込みの失敗場合
---------------------------------------------------------------------------
下記はファインチューニング用のモデルを読み込みに失敗したときのエラーログになります。 ::

   File "train_ms.py", line 303, in <module>
      main()
   File "train_ms.py", line 53, in main
      mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 200, in spawn
      return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 158, in start_processes
      while not context.join():
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 119, in join
      raise Exception(msg)
   Exception: 

   -- Process 0 terminated with the following error:
   Traceback (most recent call last):
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 20, in _wrap
      fn(i, *args)
   File "/content/drive/MyDrive/MMVC_Trainer-main/train_ms.py", line 108, in run
      _, _, _, epoch_str = utils.load_checkpoint(hps.fine_model_g, net_g, optim_g)
   File "/content/drive/MyDrive/MMVC_Trainer-main/utils.py", line 19, in load_checkpoint
      assert os.path.isfile(checkpoint_path)
   AssertionError
   
| ファインチューニング用のモデルの読み込みに失敗しています。
| /finemodel 配下に finemodel用のモデルファイル(.pth)がありますか？
| (v1.1.0 では G_232000.pthとD_232000.pth)


detasetにデータが正しく配置されていない場合
---------------------------------------------------------------------------
下記はdetasetにデータが正しく配置されていないときのエラーログになります。

size mismatch for emb_g.weight:とエラーが出た場合、ほぼ確実にdetasetの配置ミスです。 ::
   
   -- Process 0 terminated with the following error:
   Traceback (most recent call last):
   File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 20, in _wrap
      fn(i, *args)
   File "/content/drive/MyDrive/MMVC_Trainer-main/train_ms.py", line 108, in run
      _, _, _, epoch_str = utils.load_checkpoint(hps.fine_model_g, net_g, optim_g)
   File "/content/drive/MyDrive/MMVC_Trainer-main/utils.py", line 38, in load_checkpoint
      model.module.load_state_dict(new_state_dict)
   File "/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py", line 1045, in load_state_dict
      self.__class__.__name__, "\n\t".join(error_msgs)))
   RuntimeError: Error(s) in loading state_dict for SynthesizerTrn:
      size mismatch for emb_g.weight: copying a param with shape torch.Size([106, 256]) from checkpoint, the shape in current model is torch.Size([104, 256]).

| datasetに正しくデータが配置されていません。
| (後で加筆します…)
最終更新:2021/04/19