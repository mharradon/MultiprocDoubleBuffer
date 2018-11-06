def make_ctype(dtype):
  # I'm surprised I can't find this in numpy or ctypes
  dtype_str = str(dtype)
  if dtype_str=="float64":
    return "d" 
  elif dtype_str=="float32":
    return "f" 
  elif dtype_str=="float16":
    return "h" 
  raise Exception(f"Unknown dtype {dtype_str}")

def double_buffer_shm(q):
  # Pulls from multiprocessing.queue in 2 processes to shared memory
  # Interleaves calls with processing going on during yield to GPU training
  # This gets unpickling and memory copy out of the critical training path for performance
  # 2 buffers are used so one can be written to while training can occur on the other without it changing
  
  success, value = q.get()
  success_pipe_parent0, success_pipe_child0 = multiprocessing.Pipe()
  success_pipe_parent1, success_pipe_child1 = multiprocessing.Pipe()

  buf0 = [np.frombuffer(multiprocessing.Array(make_ctype(x_i.dtype),x_i.size).get_obj(),dtype=x_i.dtype).reshape(x_i.shape) for x_i in value[0]], \
            [np.frombuffer(multiprocessing.Array(make_ctype(y_i.dtype),y_i.size).get_obj(),dtype=y_i.dtype).reshape(y_i.shape) for y_i in value[1]]
  buf1 = [np.frombuffer(multiprocessing.Array(make_ctype(x_i.dtype),x_i.size).get_obj(),dtype=x_i.dtype).reshape(x_i.shape) for x_i in value[0]], \
            [np.frombuffer(multiprocessing.Array(make_ctype(y_i.dtype),y_i.size).get_obj(),dtype=y_i.dtype).reshape(y_i.shape) for y_i in value[1]]

  t0 = multiprocessing.Process(target=set_vals_from_queue,args=(buf0,q,success_pipe_child0))
  t0.start()
  t1 = multiprocessing.Process(target=set_vals_from_queue,args=(buf1,q,success_pipe_child1))
  t1.start()
  yield success, value

  for half_i in count():
    t0.join()
    success0 = success_pipe_parent0.recv()
    yield success0, buf0
    
    t0 = multiprocessing.Process(target=set_vals_from_queue,args=(buf0,q,success_pipe_child0))
    t0.start()

    t1.join()
    success1 = success_pipe_parent1.recv()
    yield success1, buf1
    
    t1 = multiprocessing.Process(target=set_vals_from_queue,args=(buf1,q,success_pipe_child1))
    t1.start()

def set_vals_from_queue(batch,q,child_pipe):
  # Writes buffer to shm and sends success through pipe
  success, next_batch = q.get()
  for xOrY in range(2):
    for i in range(len(batch[xOrY])):
      batch[xOrY][i][:] = next_batch[xOrY][i]
  child_pipe.send(success)
