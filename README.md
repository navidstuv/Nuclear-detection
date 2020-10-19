# Nuclear-detection

Test_funcv2 is file for inference:




Return the initialized model by using `model = model_initialize_detector(modelType = 'spagetti-singleHead-multiscale-residual-deep',cellLoss = '', marginLoss = '',image_size = 1024, wights_path = '')` function. 

Pass [Model checkpoint](https://drive.google.com/file/d/1K7g1l3k35r3pSiseFRF2pifw261oeeFn/view?usp=sharing) to the `wights_path` argument.
Then use `proxy_map, coordinates = detector(img, model,image_size = 1024)` and pass the model from previous function to get the probability map and the coordinates of nuclei.
