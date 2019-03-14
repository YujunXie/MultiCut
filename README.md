# MultiCut

  Current multi-target tracking algorithms are based on tracking by detection strategy, using data association methods to track the detection targets frame-by-frame. How to better solve data association problem is the key point and difficulty of multi-target tracking problem. 
  This project provided a way solving long-term multi-target tracking problem in video, and proposed a tracking algorithm based on subgraph multi-cut. 
  The algorithm mainly includes two parts, target detection and data association. The target detector uses a Deformable Parts Model (DPM) to detect objects in the video(we mainly use the results of DPM for convenience).The data association uses an algorithm named subgraph multi-cut. When building the similarity measure feature between detection hypothesis, based on the appearance information, the motion information of the target is further integrated using optical flow.**

---

- Implemented in C++
- Editor: VS2015
- Environment: Win10

Before running, you need to download the data at [MOT Chanllenge Benchmark](https://motchallenge.net/data/MOT16/)

*(It's a mistake we use the absolute path in the project, so you have to change the path defined in file 'MultiCut/DataType.h')* 

![results.png](https://i.loli.net/2019/03/14/5c8a3b3c06a7d.png)
