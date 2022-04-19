「6 学習を実行します」でエラーが発生します。
=============================================
ファインチューニング用のモデルを読み込みの失敗パターン
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
| /finemodel 配下に finemodel用のモデルファイル(.pth)がありますか？(v1.1.0 では G_232000.pthとD_232000.pth)

最終更新:2021/04/19