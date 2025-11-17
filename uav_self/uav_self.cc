#include <numeric>
#include <cmath>
#include <iostream>
#include <sys/time.h>
#include "opencv2/opencv.hpp"

#include <unistd.h>
#include <assert.h>
#include <stdio.h>
#include <pthread.h>
#include "event.h"

extern "C" {
#include "apriltag_pose.h"
#include "apriltag.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"
#include "tagCustom48h12.h"
#include "tagStandard41h12.h"
#include "tagStandard52h13.h"
#include "common/getopt.h"
}

// ----------------------- header of h264 -----------------------
#include "h264encoder.h"
#include "h264decoder.h"

// ----------------------- header of UDP -----------------------
#include <string.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <queue>
#include <vector>

#include "captureSDI.h"

// ----------------------- header of mynt -----------------------
#include <opencv2/highgui/highgui.hpp>

// init can
#include <linux/can.h>
#include <linux/can/raw.h>
//----------------------- header of mac -------------------------
#include <net/if.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>


#define PI 3.141592653
#define ETH_NAME "eth0"

using namespace std;
using namespace cv;

CEvent *receivePictureEvent = NULL;
CEvent *sendPictureEvent = NULL;
CEvent *encodePicureEvent = NULL;

//gloabl variable
bool switchGlobal = true;
Mat frameGlobal;
Mat frameDetectionGlobal;

queue<vector<uchar>> queueData;
queue<Mat> queueMat;

int sizeGlobal = 0;
uchar* dataGlobal ;
int flag_show = true;

pthread_mutex_t mylock = PTHREAD_MUTEX_INITIALIZER;

CEvent *CreateEvent(bool bManualReset, bool bInitialState)
{
    CEvent *pEvent = new CEvent(bManualReset, bInitialState);
    assert(pEvent);
 
    bool bRet = pEvent->CreateEvent();
    assert(bRet);
 
    return pEvent;
}
 
unsigned int WaitForSingleObject(CEvent *pEvent, int cms)
{
    assert(pEvent);
    if( pEvent->Wait(cms) )
    {
        return 0;
    }
 
    return 1;
}
 
bool CloseHandle(CEvent *pEvent)
{
    delete pEvent;
}
 
bool SetEvent(CEvent *pEvent)
{
    pEvent->Set();
}
 
bool ResetEvent(CEvent *pEvent)
{
    pEvent->Reset();
}


void *readCamera(void *pEvent)
{
    Mat frame;
    
    //// ----------------------- v4l2 -----------------------
    //int fd;
    //fd = open("/dev/video0", O_RDWR);
    //if (fd == -1)
    //{
        //perror("Opening video device");
        ////return 1;
    //}
    //print_caps(fd);
    //init_mmap(fd);
    //// ----------------------- v4l2 -----------------------

	// ----------------------- opencv -----------------------
    VideoCapture cap;
    VideoWriter vw;
        int fourcc = vw.fourcc('M','J','P','G');
        cap.open("/dev/video0");
        if(!cap.isOpened())
		{
			cap.open("/dev/video2");
			cout << "video2\n";
		}
	if(!cap.isOpened())
		{
			cap.open("/dev/video1");
			cout << "video1\n";
		}
        cap.set(CAP_PROP_FOURCC,fourcc);
        
        cap.set(CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(CAP_PROP_FRAME_HEIGHT, 1024); 
        cap.set(CAP_PROP_FPS,30);
        cout << cap.get(CAP_PROP_FRAME_WIDTH) << " " << cap.get(CAP_PROP_FRAME_HEIGHT) << endl;
        if(!cap.isOpened())
            cout << "camera is not open! " << endl;
        cout << "CAP_PROP_CONTRAST: "<<cap.get(CAP_PROP_CONTRAST) << endl;
        //CV_CAP_PROP_CONTRAST: 2
        cap.set(CAP_PROP_CONTRAST, 2);
        cout << "CAP_PROP_CONTRAST: "<<cap.get(CAP_PROP_CONTRAST) << endl;
        // ----------------------- opencv -----------------------
    
    
    //// ----------------------- mynt -----------------------
    //Camera cam;
    //DeviceInfo dev_info;
    //util::select(cam, &dev_info);

    //util::print_stream_infos(cam, dev_info.index);

    //std::cout << "Open device: " << dev_info.index << ", "
      //<< dev_info.name << std::endl << std::endl;

    //OpenParams params(dev_info.index);
    //{
        //// Framerate: 10(default usb3.0) 5(default usb2.0), [0,60], [30](STREAM_2560x720)
        //params.framerate = 30;

        //// Device mode, default DEVICE_ALL
        //// params.dev_mode = DeviceMode::DEVICE_ALL;

        //// Color mode: raw(default), rectified
        //// params.color_mode = ColorMode::COLOR_RECTIFIED;

        //// Stream mode: left color only
        //params.stream_mode = StreamMode::STREAM_640x480;  // vga
        ////params.stream_mode = StreamMode::STREAM_1280x720;  // hd
        //// Stream mode: left+right color
        //// params.stream_mode = StreamMode::STREAM_1280x480;  // vga
        //// params.stream_mode = StreamMode::STREAM_2560x720;  // hd

        //// Auto-exposure: true(default), false
        //// params.state_ae = false;

        //// Auto-white balance: true(default), false
        //// params.state_awb = false;

        //// IR Depth Only: true, false(default)
        //// Note: IR Depth Only mode support frame rate between 15fps and 30fps.
        ////     When dev_mode != DeviceMode::DEVICE_ALL,
        ////       IR Depth Only mode not be supported.
        ////     When stream_mode == StreamMode::STREAM_2560x720,
        ////       frame rate only be 15fps in this mode.
        ////     When frame rate less than 15fps or greater than 30fps,
        ////       IR Depth Only mode will be not available.
        //// params.ir_depth_only = true;

        //// Infrared intensity: 0(default), [0,10]
        //params.ir_intensity = 0;

        //// Colour depth image, default 5000. [0, 16384]
        //params.colour_depth_value = 5000;
    //}

    //// Enable what process logics
    //// cam.EnableProcessMode(ProcessMode::PROC_IMU_ALL);

    //// Enable image infos
    //cam.EnableImageInfo(true);

    //cam.Open(params);

    ////std::cout <<"------------------" << std::endl;
    //if (!cam.IsOpened()) {
    //std::cerr << "Error: Open camera failed" << std::endl;
        ////return 1;
    //}
    //std::cout << "Open device success" << std::endl << std::endl;

    //std::cout << "Press ESC/Q on Windows to terminate" << std::endl;

    //bool is_left_ok = cam.IsStreamDataEnabled(ImageType::IMAGE_LEFT_COLOR);
    ////bool is_depth_ok = cam.IsStreamDataEnabled(ImageType::IMAGE_DEPTH);
    //bool is_depth_ok = false;
    

    //CVPainter painter;
    //util::Counter counter(params.framerate);
    //// ----------------------- mynt -----------------------
    
    
    
    
    
	struct timeval tv_begin1, tv_end1;
	gettimeofday(&tv_begin1, NULL);
	//clock_t wbegin = clock();
    int num = 0;
    double start = cv::getTickCount();
    while (1)
    {
    
        //cout << endl << "Get image from camera." << endl;
        double end = (cv::getTickCount() - start) *1000/(getTickFrequency());
        num ++ ; 
        if(end >1000)
        {
            cout << endl << "--------------------countimg---------------------" << num << endl;
            start = cv::getTickCount();
            num = 0;
        }
        
        cap >> frame;
        //capture_image(fd,frame);
        //cam.WaitForStream();
        //cout << frame.cols << frame.rows << endl;
        
        //auto allow_count = false;
        //if (is_left_ok) {
            //auto left_color = cam.GetStreamData(ImageType::IMAGE_LEFT_COLOR);
            //if (left_color.img) {
                //allow_count = true;
                ////cv::Mat left = left_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
                //frame = left_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
                //painter.DrawSize(frame, CVPainter::TOP_LEFT);
                //painter.DrawStreamData(frame, left_color, CVPainter::TOP_RIGHT);
                //painter.DrawInformation(frame, util::to_string(counter.fps()),
                //CVPainter::BOTTOM_RIGHT);
                ////cv::imshow("left color", left);
            //}
        //}
        
        if(frame.empty())
        {   
            break;
        }
        frameGlobal = frame.clone();
        // After get img set signal as true
        //------------------------------------ set signal ------------------------------------
        SetEvent(receivePictureEvent);

		gettimeofday(&tv_end1, NULL);
		
		double timeUsed1 = (tv_end1.tv_sec - tv_begin1.tv_sec) *1000 + double((tv_end1.tv_usec - tv_begin1.tv_usec) / 10000);

		usleep(20000);
        //waitKey(30);
    }
    //cap.release();

}


struct tag_info
{
  int id;
  double tagSize;
  double x;
  double y;
  double z;
  int yaw;  
};

struct final_tag_info
{
  double c_x;
  double c_y;
  double c_z;
  int yaw;  
};



final_tag_info pos_output(vector <tag_info> V_tagInfo)//Process the original information and output the result of positioning
{
    final_tag_info f_t_i;

    for (int i = 0; i < V_tagInfo.size(); i++)
    {
        //Only one Tag is detected. There are four possibilities.
        //1.1 id=4 size=30*30 bias:x=-24 y=0
        //1.2 id=5 size=30*30 bias:x=+24 y=0
        //1.3 id=2 size=10*10 bias:x=0 y=0
        //1.4 id=3 size=10*10 bias:x=0 y=+10

        //1.1 id=4 size=20*20 bias:x=-20 y=5
        //1.2 id=5 size=20*20 bias:x=+20 y=5
        //1.3 id=2 size=10*10 bias:x=0 y=0
        //1.4 id=3 size=10*10 bias:x=0 y=+12

        if(V_tagInfo.size() == 1 && V_tagInfo[i].id == 4)
        {
            cout << "tagID=4!!!" << endl;
            f_t_i.c_x = V_tagInfo[i].x + 0.20;
            f_t_i.c_y = V_tagInfo[i].y + 0.05;
            f_t_i.c_z = V_tagInfo[i].z;
            f_t_i.yaw = V_tagInfo[i].yaw;
            //cout<< "f_t_i.c_x:"<<f_t_i.c_x<<endl;
            //cout<< "f_t_i.c_y:"<<f_t_i.c_y<<endl;
            //cout<< "f_t_i.c_z:"<<f_t_i.c_z<<endl;
            
        }
        else if (V_tagInfo.size() == 1 && V_tagInfo[i].id == 5)
        {
            cout << "tagID=5!!!" << endl;
            f_t_i.c_x = V_tagInfo[i].x - 0.20;
            f_t_i.c_y = V_tagInfo[i].y + 0.05;
            f_t_i.c_z = V_tagInfo[i].z;
            f_t_i.yaw = V_tagInfo[i].yaw;    
            //cout<< "f_t_i.c_x:"<<f_t_i.c_x<<endl;
            //cout<< "f_t_i.c_y:"<<f_t_i.c_y<<endl;
            //cout<< "f_t_i.c_z:"<<f_t_i.c_z<<endl;
        }
         else if (V_tagInfo.size() == 1 && V_tagInfo[i].id == 2)
        {
            cout << "tagID=2!!!" << endl;
            f_t_i.c_x = V_tagInfo[i].x;
            f_t_i.c_y = V_tagInfo[i].y;
            f_t_i.c_z = V_tagInfo[i].z;
            f_t_i.yaw = V_tagInfo[i].yaw;
            //cout<< "f_t_i.c_x:"<<f_t_i.c_x<<endl;
            //cout<< "f_t_i.c_y:"<<f_t_i.c_y<<endl;
            //cout<< "f_t_i.c_z:"<<f_t_i.c_z<<endl;
            
        }
         else if (V_tagInfo.size() == 1 && V_tagInfo[i].id == 3)
        {
            cout << "tagID=3!!!" << endl;
            f_t_i.c_x = V_tagInfo[i].x;
            f_t_i.c_y = V_tagInfo[i].y + 0.12;
            f_t_i.c_z = V_tagInfo[i].z;
            f_t_i.yaw = V_tagInfo[i].yaw;
            //cout<< "f_t_i.c_x:"<<f_t_i.c_x<<endl;
            //cout<< "f_t_i.c_y:"<<f_t_i.c_y<<endl;
            //cout<< "f_t_i.c_z:"<<f_t_i.c_z<<endl;
        }
        //Two Tags are detected. There are six possibilities.
        //2.1 id=2+3 bias:(x2+x3)/2 (y2+y3+10)/2
        //2.2 id=2+4 bias:(x2+x4+24)/2 (y2+y4+0)/2
        //2.3 id=2+5 bias:(x2+x5-10)/2 (y2+y5+10)/2
        //2.4 id=3+4 bias:(x3+x4+10)/2 (y3+y4-10)/2
        //2.5 id=3+5 bias:(x3+x5-10)/2 (y3+y5-10)/2
        //2.6 id=4+5 bias:(x4+x5)/2 (y4+y5)/2
         else if (V_tagInfo.size() == 2)
        {
            if ((V_tagInfo[0].id==2 && V_tagInfo[1].id==3) || (V_tagInfo[0].id==3 && V_tagInfo[1].id==2))
            {
                cout << "tagID=2+3!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x)/2;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + 0.12)/2;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z)/2;
                f_t_i.yaw = V_tagInfo[0].yaw;
                cout<< "f_t_i.c_x:"<<f_t_i.c_x<<endl;
                cout<< "f_t_i.c_y:"<<f_t_i.c_y<<endl;
                cout<< "f_t_i.c_z:"<<f_t_i.c_z<<endl;
            }
            else if ((V_tagInfo[0].id==2 && V_tagInfo[1].id==4) || (V_tagInfo[0].id==4 && V_tagInfo[1].id==2))
            {
                cout << "tagID=2+4!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x + 0.20)/2;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + 0.05)/2;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z)/2;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }
            else if ((V_tagInfo[0].id==2 && V_tagInfo[1].id==5) || (V_tagInfo[0].id==5 && V_tagInfo[1].id==2))
            {
                cout << "tagID=2+5!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x - 0.20)/2;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + 0.05)/2;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z)/2;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }
            else if ((V_tagInfo[0].id==3 && V_tagInfo[1].id==4) || (V_tagInfo[0].id==4 && V_tagInfo[1].id==3))
            {
                cout << "tagID=3+4!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x + 0.20)/2;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + 0.17)/2;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z)/2;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }
            else if ((V_tagInfo[0].id==3 && V_tagInfo[1].id==5) || (V_tagInfo[0].id==5 && V_tagInfo[1].id==3))
            {
                cout << "tagID=3+5!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x - 0.20)/2;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + 0.17)/2;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z)/2;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }
            else if ((V_tagInfo[0].id==4 && V_tagInfo[1].id==5) || (V_tagInfo[0].id==5 && V_tagInfo[1].id==4))
            {
                cout << "tagID=4+5!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x)/2;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + 0.1 )/2;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z)/2;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }           
        }
        //Three Tags are detected. There are four possibilities.
        //3.1 id=2+3+4 bias:(x2+x3+x4+10)/3 (y2+y3+y4)/3
        //3.2 id=2+3+5 bias:(x2+x4+x5-10)/3 (y2+y4+y5)/3
        //3.3 id=3+4+5 bias:(x2+x5+x4)/3 (y3+y4+y5-10)/3
        //3.4 id=2+4+5 bias:(x2+x4+x5)/3 (y2+y4+y5+10)/3
        else if (V_tagInfo.size() == 3)
        {

            if ( (V_tagInfo[0].id==2 && V_tagInfo[1].id==3 && V_tagInfo[2].id==4) || 
                 (V_tagInfo[0].id==2 && V_tagInfo[1].id==4 && V_tagInfo[2].id==3) ||
                 (V_tagInfo[0].id==3 && V_tagInfo[1].id==2 && V_tagInfo[2].id==4) ||
                 (V_tagInfo[0].id==3 && V_tagInfo[1].id==4 && V_tagInfo[2].id==2) ||
                 (V_tagInfo[0].id==4 && V_tagInfo[1].id==2 && V_tagInfo[2].id==3) ||
                 (V_tagInfo[0].id==4 && V_tagInfo[1].id==3 && V_tagInfo[2].id==2) )
            {
                //cout << "tagID=2+3+4!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x + V_tagInfo[2].x + 0.20)/3;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + V_tagInfo[2].y + 0.17)/3;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z + V_tagInfo[2].z)/3;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }
            else if ( (V_tagInfo[0].id==2 && V_tagInfo[1].id==3 && V_tagInfo[2].id==5) || 
                 (V_tagInfo[0].id==2 && V_tagInfo[1].id==5 && V_tagInfo[2].id==3) ||
                 (V_tagInfo[0].id==3 && V_tagInfo[1].id==2 && V_tagInfo[2].id==5) ||
                 (V_tagInfo[0].id==3 && V_tagInfo[1].id==5 && V_tagInfo[2].id==2) ||
                 (V_tagInfo[0].id==5 && V_tagInfo[1].id==2 && V_tagInfo[2].id==3) ||
                 (V_tagInfo[0].id==5 && V_tagInfo[1].id==3 && V_tagInfo[2].id==2) )
            {
                //cout << "tagID=2+3+5!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x + V_tagInfo[2].x - 0.20)/3;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + V_tagInfo[2].y + 0.17)/3;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z + V_tagInfo[2].z)/3;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }
            else if ( (V_tagInfo[0].id==4 && V_tagInfo[1].id==3 && V_tagInfo[2].id==5) || 
                 (V_tagInfo[0].id==4 && V_tagInfo[1].id==5 && V_tagInfo[2].id==3) ||
                 (V_tagInfo[0].id==3 && V_tagInfo[1].id==4 && V_tagInfo[2].id==5) ||
                 (V_tagInfo[0].id==3 && V_tagInfo[1].id==5 && V_tagInfo[2].id==4) ||
                 (V_tagInfo[0].id==5 && V_tagInfo[1].id==4 && V_tagInfo[2].id==3) ||
                 (V_tagInfo[0].id==5 && V_tagInfo[1].id==3 && V_tagInfo[2].id==4)  )
            {
                //cout << "tagID=4+3+5!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x + V_tagInfo[2].x)/3;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + V_tagInfo[2].y + 0.22)/3;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z + V_tagInfo[2].z)/3;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }
            else if ( (V_tagInfo[0].id==4 && V_tagInfo[1].id==2 && V_tagInfo[2].id==5) || 
                 (V_tagInfo[0].id==4 && V_tagInfo[1].id==5 && V_tagInfo[2].id==2) ||
                 (V_tagInfo[0].id==2 && V_tagInfo[1].id==4 && V_tagInfo[2].id==5) ||
                 (V_tagInfo[0].id==2 && V_tagInfo[1].id==5 && V_tagInfo[2].id==4) ||
                 (V_tagInfo[0].id==5 && V_tagInfo[1].id==4 && V_tagInfo[2].id==2) ||
                 (V_tagInfo[0].id==5 && V_tagInfo[1].id==2 && V_tagInfo[2].id==4)  )
            {
               // cout << "tagID=2+4+5!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x + V_tagInfo[2].x)/3;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + V_tagInfo[2].y + 0.1)/3;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z + V_tagInfo[2].z)/3;
                f_t_i.yaw = V_tagInfo[0].yaw;
            }
        }
        //Four Tags are detected. There is one possibility.
        //4.1 id=2+3+4+5 bias:(x2+x3+x4+x5)/4 (y2+y3+y4+y5)/4
        else if (V_tagInfo.size() == 4)
        {
                //cout << "tagID=2+3+4+5!!!" << endl;
                f_t_i.c_x = (V_tagInfo[0].x + V_tagInfo[1].x + V_tagInfo[2].x + V_tagInfo[3].x)/4;
                f_t_i.c_y = (V_tagInfo[0].y + V_tagInfo[1].y + V_tagInfo[2].y + V_tagInfo[3].y + 0.22)/4;
                f_t_i.c_z = (V_tagInfo[0].z + V_tagInfo[1].z + V_tagInfo[2].z + V_tagInfo[3].z)/4;
                f_t_i.yaw = V_tagInfo[0].yaw;
        }
    }
    
    return f_t_i;
    
}

void *detection(void *pEvent)
{
	getopt_t *getopt = getopt_create();
	VideoWriter video("/home/hyq/Downloads/measurement/res.mp4", VideoWriter::fourcc('D','I','V','X'),30.0, Size(960,540));
	getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
	getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
	getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
	getopt_add_string(getopt, 'f', "family", "tag36h11", "Tag family to use");
	getopt_add_int(getopt, 't', "threads", "1", "Use this many CPU threads");
	getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
	getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
	getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");
    
	

	
    //apriltag_detection_info_t info; 
	
	//info.tagsize = 0.665;
	//info.tagsize = 0.132; 
   // info.tagsize = 0.33;
    
    //960*540
	//info.fx = 758.9956455647;     
	//info.fy = 759.576308726;
	//info.cx = 459.695841775;     
    //info.cy = 259.0498361309;

	//1920*1080
	//info.tagsize = 0.665; 
    /*info.tagsize = 0.135;
	info.fx = 1399.008608;     
	info.fy = 1398.06775013;
	info.cx = 912.8436694988;     
	info.cy = 560.97156462088;*/
    //xiao mi nei can
	//640-480
	//info.fx = 373.954095;    
	//info.fy = 380.075023;
	//info.cx = 323.721185;    
	//info.cy = 228.173866;
	//1280-720
	
	//info.fx = 776.988440;
	//info.fy = 775.932595;
	//info.cx = 692.768890;
	//info.cy = 370.918443;
    
	// Initialize tag detector with options
	apriltag_family_t *tf = NULL;
	const char *famname = getopt_get_string(getopt, "family");
	if (!strcmp(famname, "tag36h11")) {
		tf = tag36h11_create();
	}
	else if (!strcmp(famname, "tag25h9")) {
		tf = tag25h9_create();
	}
	else if (!strcmp(famname, "tag16h5")) {
		tf = tag16h5_create();
	}
	else if (!strcmp(famname, "tagCircle21h7")) {
		tf = tagCircle21h7_create();
	}
	else if (!strcmp(famname, "tagCircle49h12")) {
		tf = tagCircle49h12_create();
	}
	else if (!strcmp(famname, "tagStandard41h12")) {
		tf = tagStandard41h12_create();
	}
	else if (!strcmp(famname, "tagStandard52h13")) {
		tf = tagStandard52h13_create();
	}
	else if (!strcmp(famname, "tagCustom48h12")) {
		tf = tagCustom48h12_create();
	}
	else {
		printf("Unrecognized tag family name. Use e.g. \"tag36h11\".\n");
		exit(-1);
	}


	apriltag_detector_t *td = apriltag_detector_create();
	apriltag_detector_add_family(td, tf);
	
	td->quad_decimate = getopt_get_double(getopt, "decimate");
	td->quad_sigma = getopt_get_double(getopt, "blur");
	td->nthreads = getopt_get_int(getopt, "threads");
	td->debug = getopt_get_bool(getopt, "debug");
	td->refine_edges = getopt_get_bool(getopt, "refine-edges");
	Mat gray;

	
    int NN=0;
	
	int fps = 0;
    int count = 0;
    int count1=0;
    double start = cv::getTickCount();
    
    //  ---------------------------send UDP init ---------------------------
    #define DEST_PORT_CONTROL 6083
    #define DSET_IP_ADDRESS  "226.0.0.80"
    
    int sock_fd;

    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock_fd < 0)
    {
        perror("socket");
        exit(1);
    }

    struct sockaddr_in addr_serv;
    int len;
    memset(&addr_serv, 0, sizeof(addr_serv));
    addr_serv.sin_family = AF_INET;
    addr_serv.sin_addr.s_addr = inet_addr(DSET_IP_ADDRESS);
    addr_serv.sin_port = htons(DEST_PORT_CONTROL);
    len = sizeof(addr_serv);;

    uchar send_buf[20];
    send_buf[0] = 0xEB;
	send_buf[1] = 0x90;
	send_buf[2] = 0x11;
	send_buf[3] = 0x10;
	send_buf[4] = 0x00;
	//x:5,6 y:78, z:9,10, angle:11,12 ,13,14,15,16 flag
	send_buf[17] = 0x00;
	send_buf[18] = 0x00;
	send_buf[19] = 0x00;
	//  ---------------------------send UDP init finish ---------------------------
	
	//  ---------------------------CAN init ---------------------------
    int s;
    int nbytes;
    struct sockaddr_can addr;
    struct ifreq ifr;

    const char *ifname = "can1";

    if((s = socket(PF_CAN, SOCK_RAW, CAN_RAW)) < 0) 
    {
        perror("Error while opening socket");
        //return -1;
    }

    strcpy(ifr.ifr_name, ifname);
    ioctl(s, SIOCGIFINDEX, &ifr);
    
    addr.can_family  = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    printf("%s at index %d\n", ifname, ifr.ifr_ifindex);

    if(bind(s, (struct sockaddr *)&addr, sizeof(addr)) < 0) 
    {
        perror("Error in socket bind");
        //return -2;
    }
    //  ---------------------------CAN init finish---------------------------
    
    while(1)
    {
        //init send_buf
        for(int i = 5; i < 20; i++)
		    send_buf[i] = 0x00;
		    
		//init can
		/**
         * struct can_frame - basic CAN frame structure
         * @can_id:  CAN ID of the frame and CAN_*_FLAG flags, see canid_t definition
         * @can_dlc: frame payload length in byte (0 .. 8) aka data length code
         *           N.B. the DLC field from ISO 11898-1 Chapter 8.4.2.3 has a 1:1
         *           mapping of the 'data length code' to the real payload length
         * @__pad:   padding
         * @__res0:  reserved / padding
         * @__res1:  reserved / padding
         * @data:    CAN frame payload (up to 8 byte)
         */
    
        // init struct frame all 0
        struct can_frame canframe;
        //frame.can_id = frame.can_id & CAN_EFF_MASK;
        memset(&canframe, 0, sizeof(struct can_frame));
        canframe.can_id = CAN_EFF_FLAG | (200 << 16);
        canframe.can_id = canframe.can_id + (10 << 8);
        canframe.can_dlc = 8;
        // flag : angle, z, y, x
        // 0 invalid
        // 1 vaild
        // angle <<7 , z <<6 , y <<5 , x << 4
        
        //sign  : angle, z, y, x
        // 0 plus  sign
        // 1 minus sign
        // angle always = 0
        // angle << 3, z << 2, y << 1,  x
        
        WaitForSingleObject(receivePictureEvent, INFINITE);
        ResetEvent(receivePictureEvent);
        double end = (cv::getTickCount() - start) *1000/(getTickFrequency());
        count1 ++ ; 
        if(end >1000)
        {
            cout << "--------------------countdetection---------------" << count1 << endl;
            start = cv::getTickCount();
            count1 = 0;
        }
		Mat frame = frameGlobal.clone();
		pthread_mutex_unlock(&mylock);
		NN++;
		
		int width = frame.cols;
		int height = frame.rows;
		cout << "--------------------width---------------" << width << "--------------------height---------------" << height << endl;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		
		
		// Make an image_u8_t header for the Mat data
		//image_u8_t im = image_u8_t(gray.cols, gray.rows, gray.cols, gray.data);
		image_u8_t im = 
		{gray.cols,
			gray.rows,
			gray.cols,
			gray.data
		};
		zarray_t *detections = apriltag_detector_detect(td, &im);
		
        //cout << zarray_size(detections) << " tags detected" << endl;

		double local[2][2];    
		double locar[2][2];

		apriltag_pose_t pose;
        int x = 0;
        int y = 0;
        int z = 0;
        int int_yaw = 0;
        int flag_x = 0;
        int flag_y = 0;
        int flag_z = 0;
        int flag_yaw = 0;
        tag_info t_i;
        vector<tag_info> V_tagInfo;
		// Draw detection outlines
		for (int i = 0; i < zarray_size(detections); i++) 
        {
            
            
			apriltag_detection_t *det;
			zarray_get(detections, i, &det); 

			line(frame, Point(det->p[0][0], det->p[0][1]),
				Point(det->p[1][0], det->p[1][1]),
				Scalar(0, 0xff, 0), 2);
			line(frame, Point(det->p[0][0], det->p[0][1]),
				Point(det->p[3][0], det->p[3][1]),
				Scalar(0, 0, 0xff), 2);
			line(frame, Point(det->p[1][0], det->p[1][1]),
				Point(det->p[2][0], det->p[2][1]),
				Scalar(0xff, 0, 0), 2);
			line(frame, Point(det->p[2][0], det->p[2][1]),
				Point(det->p[3][0], det->p[3][1]),
				Scalar(0xff, 0, 0), 2);


			//cout << "pixel offset(pix):" << "x= " << det->p[3][0] << "  " << "y= " << det->p[3][1] << endl;
			local[i][0] = det->p[3][0];
			local[i][1] = det->p[3][1];
			locar[i][0] = det->p[2][0];
			locar[i][1] = det->p[2][1];
			
            
            //get tagsize 
            int tmp_id =det->id;
            double tmp_tagsize = 0;
            t_i.id = det->id;
            if (det->id == 2 || det->id == 3)
            {
                tmp_tagsize = 0.1;
                t_i.tagSize = tmp_tagsize;
            }
            else if (det->id == 4 || det->id == 5)
            {
                tmp_tagsize = 0.2;
                t_i.tagSize = tmp_tagsize;
            }
           
            
            
            
            
             apriltag_detection_info_t info; 
             info.tagsize = tmp_tagsize;
			 info.fx = 333.225420;     
			 info.fy = 333.372990;
			 info.cx = 338.694728;   
			 info.cy = 212.187588;
			 //info.fx = 340.57432691;     
			 //info.fy = 343.16692839;
			 //info.cx = 328.19849198;   
			 //info.cy = 246.52242198;
            
            
            
            
			info.det = det;  //det-->info

			estimate_pose_for_tag_homography(&info, &pose);
			
			//cout << "distance offset(cm):" << "x= " << pose.t->data[0]*100 << "  " << "y= " << pose.t->data[1]*100 << " " << "z= " << pose.t->data[2]*100 << endl;  //t output
			/************output the information of yaw***************/
			
			double center_up_x = det->p[3][0] + (det->p[2][0] - det->p[3][0]) / 2;
			double center_up_y = det->p[3][1] + (det->p[2][1] - det->p[3][1]) / 2;
			//cout << "center of TAG:" << "c1= " << det->c[0] << "  " << "c2= " << det->c[1]  << endl;  
			//cout << "center of Top:" << "c1= " << center_up_x << "  " << "c2= " << center_up_y << endl;  
			float vector1x = det->c[0] - center_up_x;
			float vector1y = det->c[1] - center_up_y;
			float vector2x = 0 - 0;
			float vector2y = 10 - 0;
			float jiaodu_temp = ((vector1x)*(vector2x)+(vector1y)*(vector2y)) / (sqrt(pow(vector1x, 2) + pow(vector1y, 2))*sqrt(pow(vector2x, 2) + pow(vector2y, 2)));
			float yaw =  acos(jiaodu_temp)*(180 / PI);
			//cout << "yaw of uav:" << "yaw= " << yaw << "jiaodu" << endl;  //t output
            //direction
            int directionFlag = vector1x * vector2y - vector1y * vector2x;
           // String true_yaw;
            //int flag_yaw;
            if(directionFlag < 0)
            {
               // cout << "left---------------!!!!!!!!!!!!!!!!!" << endl;
               // true_yaw = "-" + to_string(yaw);
                 //cout << "yaw of uav:" << "yaw= " << "-" << yaw << " jiaodu" << endl;
				 flag_yaw =1;	
                
            }
            else if(directionFlag > 0) 
            {
               // cout << "right--------------+++++++++++++++++ " << endl;
                 //true_yaw = "+" + to_string(yaw);
                   //cout << "yaw of uav:" << "yaw= " << "+" << yaw << " jiaodu" << endl;
                   flag_yaw=2;
            }
            if(flag_yaw == 2)
            {
                int_yaw = 10*abs(yaw);
            }
            else if(flag_yaw == 1)
            {
                int_yaw = 10*(360-yaw);
            }
			flag_yaw =1 ;
            
            t_i.yaw=int_yaw;
            t_i.x =pose.t->data[0];
            t_i.y =pose.t->data[1];  
            t_i.z =pose.t->data[2];
            
            V_tagInfo.push_back(t_i);
        }
            
        if(zarray_size(detections) > 0)
        {    // ------------
            final_tag_info ff_t_i = pos_output(V_tagInfo);
            V_tagInfo.clear();
            
            //cout << "out--ff_t_i.c_x:" << ff_t_i.c_x << endl;
            //cout << "out--ff_t_i.c_y:" << ff_t_i.c_y << endl;			
			//x:5,6 y:78, z:9,10, angle:11,12 ,13,14,15,16 flag
			x = abs(ff_t_i.c_x*1000);
			y = abs(ff_t_i.c_y*1000);
			z = abs(ff_t_i.c_z*1000);
            //cout << "out--xxxxxxxxxxxxx:" << x << endl;
            //cout << "out--yyyyyyyyyyyyy:" << y << endl;
			//int_yaw = abs(yaw);
			//int_yaw = abs(yaw * 10);
			
			// 0 can not use 1: zheng 2 fu
			//int flag_x = 0;
			//int flag_y = 0;
			//int flag_z = 0;
            
            
			if(ff_t_i.c_x<0)
				flag_x = 2;
			else if(ff_t_i.c_x>0)
				flag_x = 1;
			if(ff_t_i.c_y<0)
				flag_y = 1;
			else if(ff_t_i.c_y>0)
				flag_y = 2;
			if(ff_t_i.c_z<0)
				flag_z = 2;
			else if(ff_t_i.c_z>0)
				flag_z = 1;
            
            			
			//x:5,6 y:78, z:9,10, angle:11,12 ,13,14,15,16 flag
			//x = abs(pose.t->data[0]*1000);
			//y = abs(pose.t->data[1]*1000);
			//z = abs(pose.t->data[2]*1000);
			//int_yaw = abs(yaw);
			//int_yaw = abs(yaw * 10);
			
			// 0 can not use 1: zheng 2 fu
			//int flag_x = 0;
			//int flag_y = 0;
			//int flag_z = 0;
            
            /*
			if(pose.t->data[0]<0)
				flag_x = 2;
			else if(pose.t->data[0]>0)
				flag_x = 1;
			if(pose.t->data[1]<0)
				flag_y = 1;
			else if(pose.t->data[1]>0)
				flag_y = 2;
			if(pose.t->data[2]<0)
				flag_z = 2;
			else if(pose.t->data[2]>0)
				flag_z = 1;
            */
			
            
            
			send_buf[5] = (uchar)(0x00ff & x);
			send_buf[6] = (uchar)((0xff00 & x) >> 8);
			send_buf[7] = (uchar)(0x00ff & y);
			send_buf[8] = (uchar)((0xff00 & y) >> 8); 
			send_buf[9] = (uchar)(0x00ff & z);
			send_buf[10] = (uchar)((0xff00 & z) >> 8);
			send_buf[11] = (uchar)(0x00ff & ff_t_i.yaw);
			send_buf[12] = (uchar)((0xff00 & ff_t_i.yaw) >> 8);
			send_buf[13] = (uchar)(0x00ff & flag_x);
			send_buf[14] = (uchar)(0x00ff & flag_y);
			send_buf[15] = (uchar)(0x00ff & flag_z);
			send_buf[16] = (uchar)(0x00ff & flag_yaw);
			// 0 can not use 1: zheng 2 fu
		    // flag : angle, z, y, x
            // 0 invalid
            // 1 vaild
            // angle <<7 , z <<6 , y <<5 , x << 4
            
            //sign  : angle, z, y, x
            // 0 plus  sign
            // 1 minus sign
            // angle always = 0
            // angle << 3, z << 2, y << 1,  x
            //cout << flag_x << " " << flag_y << " " << flag_z << endl;
            if(flag_x!=0)
                canframe.can_id = canframe.can_id + (1 << 4);
            if(flag_y!=0)
                canframe.can_id = canframe.can_id + (1 << 5);
            if(flag_z!=0)
                canframe.can_id = canframe.can_id + (1 << 6);
            if(flag_yaw!=0)
                canframe.can_id = canframe.can_id + (1 << 7);
            
            if(flag_x == 2)
                canframe.can_id = canframe.can_id + 1;
            if(flag_y == 2 )
                canframe.can_id = canframe.can_id + (1 << 1);
            if(flag_z == 2 )
                canframe.can_id = canframe.can_id + (1 << 2);
            
            canframe.data[0] = (uchar)(0x00ff & x);
			canframe.data[1] = (uchar)((0xff00 & x) >> 8);
			canframe.data[2] = (uchar)(0x00ff & y);
			canframe.data[3] = (uchar)((0xff00 & y) >> 8); 
			canframe.data[4] = (uchar)(0x00ff & z);
			canframe.data[5] = (uchar)((0xff00 & z) >> 8);
			canframe.data[6] = (uchar)(0x00ff & ff_t_i.yaw);
			canframe.data[7] = (uchar)((0xff00 & ff_t_i.yaw) >> 8);
            
            
        }    
        
        
		if(zarray_size(detections) == 0)
		{
		    cout << "No image detect!"<< endl;
			apriltag_detections_destroy(detections);
			putText(frame,"o tag",Point(50,60),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);		
		}
		else
		{
			apriltag_detections_destroy(detections);
			//putText(frame,"x: "+to_string(pose.t->data[0]*100)+ " " + "y: "+to_string(pose.t->data[1]*-100) + " " + "z: "+to_string(pose.t->data[2]*100) ,Point(50,60),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
            string xtext;
            string ytext;
            string ztext;
            string jtext;
            if(flag_x == 2)
                xtext = "x: "+to_string(x*-1);
            else
                xtext = "x: "+to_string(x);
            if(flag_y == 2) 
                ytext = "y: "+to_string(y*-1);
            else
                ytext = "y: "+to_string(y);
            if(flag_z == 2)
                ztext = "z: "+to_string(z*-1);
            else
                ztext = "z: "+to_string(z);
            if(flag_yaw == 2)
                jtext = "j: "+to_string(int_yaw*-1);
            else
                jtext = "j: "+to_string(int_yaw);
            int startx = 10;
            int starty = 50;
            int baselinex = 0;
            int baseliney = 0;
            int baselinez = 0;
            
            putText(frame, xtext, Point(startx, starty),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
            Size textSizex = getTextSize(xtext,FONT_HERSHEY_SIMPLEX, 1, 4, &baselinex);
            
            putText(frame, ytext, Point(startx, starty+textSizex.height+baselinex),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
            Size textSizey = getTextSize(ytext,FONT_HERSHEY_SIMPLEX, 1, 4, &baseliney);
            
            putText(frame, ztext, Point(startx, starty+textSizex.height+baselinex + textSizey.height + baseliney),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
            Size textSizez = getTextSize(ztext,FONT_HERSHEY_SIMPLEX, 1, 4, &baselinez);
            
            putText(frame, jtext, Point(startx, starty+textSizex.height+baselinex + textSizey.height + baseliney + textSizez.height + baselinez),FONT_HERSHEY_SIMPLEX,1,Scalar(255,23,0),4,8);
            
        }
        if(flag_show)
        {
            imshow("Tag Detections", frame);
            if (waitKey(1) == 27)
                break;
		}
        //frameDetectionGlobal = frame.clone();
        queueMat.push(frame.clone());
        SetEvent(sendPictureEvent);
        
        for(int i = 0; i < 19; i++)
            send_buf[19] = (0x00ff & ((int)send_buf[i] + send_buf[19]));
            
        int send_num = sendto(sock_fd, send_buf, sizeof(send_buf), 0, (struct sockaddr *)&addr_serv, len);

        if ((nbytes = write(s, &canframe, sizeof(canframe))) < 0)
            perror("Send error!");
        cout << "send_num feikong can : " <<nbytes << endl;
        cout << "send_num feikong net : " <<send_num << endl;
        cout << hex <<canframe.can_id << dec << endl;
        
        /*
        for(int i = 0 ; i<20 ; i++)
            cout << hex << (int)send_buf[i] << " ";
        cout << endl;
        */
        //usleep(10);

	}
	destroyAllWindows();
	//video.release();

		apriltag_detector_destroy(td);

	if (!strcmp(famname, "tag36h11")) {
		tag36h11_destroy(tf);
	}
	else if (!strcmp(famname, "tag25h9")) {
		tag25h9_destroy(tf);
	}
	else if (!strcmp(famname, "tag16h5")) {
		tag16h5_destroy(tf);
	}
	else if (!strcmp(famname, "tagCircle21h7")) {
		tagCircle21h7_destroy(tf);
	}
	else if (!strcmp(famname, "tagCircle49h12")) {
		tagCircle49h12_destroy(tf);
	}
	else if (!strcmp(famname, "tagStandard41h12")) {
		tagStandard41h12_destroy(tf);
	}
	else if (!strcmp(famname, "tagStandard52h13")) {
		tagStandard52h13_destroy(tf);
	}
	else if (!strcmp(famname, "tagCustom48h12")) {
		tagCustom48h12_destroy(tf);
	}


	getopt_destroy(getopt);
}


void *encodeImg(void *pEvent)
{   
    //init 
    cv::Mat frame;
    cv::Mat dst;

    cv::VideoCapture videoCapture("/home/hyq/Downloads/square.mp4");
    
    h264Encoder h264;
    AvH264EncConfig conf;
    conf.width = 640;
    conf.height = 480;
    //conf.width = 1920;
    //conf.height = 1080;
    // frame I, I the less, the smaller
    conf.gop_size = 3;
    conf.max_b_frames = 0;
    conf.frame_rate = 30;
    //conf.bit_rate = 320000;
    conf.bit_rate = 1024*1024*2;
    h264.Init(conf);

    cv::Mat cvDst;
    int nWaitTime =1;
    int count = 0;
    double start = cv::getTickCount();
    
    while (switchGlobal)
    {
        WaitForSingleObject(sendPictureEvent, INFINITE);
        ResetEvent(sendPictureEvent);
        //frame = frameDetectionGlobal.clone();
        
        //videoCapture >>frame;
        if(queueMat.size()==0)
            continue;
        frame = queueMat.front();

        double end = (cv::getTickCount() - start) *1000/(getTickFrequency());
        count ++ ; 
        if(end >1000)
        {
            cout << "--------------------countencode------------------" << count << endl;
            start = cv::getTickCount();
            count = 0;
        }
        // 开始计时
        double timePoint1 = cv::getTickCount();
        
        
        
        if( frame.empty())
        {
            cout << "empty" << endl;
            continue;
        }
        
        cv::Mat _frame;
        cv::resize(frame,_frame,cv::Size(conf.width, conf.height),0,0);

        // do encode
        AVPacket *pkt = h264.encode(_frame);
        int size = pkt->size;
        uchar* data = nullptr;
        data = pkt->data;
        //dataGlobal
        dataGlobal = new uchar[size];
        memcpy(dataGlobal, data, size);
        
        // --------------test queue
        //vector<uchar> dataUchar(data, data+size);
        //queueData.push(dataUchar);
        
        sizeGlobal = size;
        queueMat.pop();
        SetEvent(encodePicureEvent);
    } 
}
void *sendUDP(void *pEvent)
{

    #define DEST_PORT 6001
    //#define DSET_IP_ADDRESS  "224.0.0.1"
    #define DSET_IP_ADDRESS  "226.0.0.80"
    
    int sock_fd;

    sock_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sock_fd < 0)
    {
        perror("socket");
        exit(1);
    }

    struct sockaddr_in addr_serv;
    int len;
    memset(&addr_serv, 0, sizeof(addr_serv));
    addr_serv.sin_family = AF_INET;
    addr_serv.sin_addr.s_addr = inet_addr(DSET_IP_ADDRESS);
    addr_serv.sin_port = htons(DEST_PORT);
    len = sizeof(addr_serv);
    int sizeUDP;
    uchar* dataUDP;
    //int stepUDP  = 4096;
    int stepUDP = 2048;
    //int stepUDP = 65507;
    int step;
    int startUDP = 0;
    int send_num;
    uchar *send_buf;
    
    double start = cv::getTickCount();
    int countUDP = 0;
    int count=0;
    while (switchGlobal)
    {
        WaitForSingleObject(encodePicureEvent, INFINITE);
        ResetEvent(encodePicureEvent);
        double end = (cv::getTickCount() - start) *1000/(getTickFrequency());
        countUDP ++ ; 
        if(end >1000)
        {
            cout << "--------------------countUDP---------------------" << countUDP << endl;
            start = cv::getTickCount();
            countUDP = 0;
        }
        
        
        double timePoint1 = cv::getTickCount();
        sizeUDP = sizeGlobal;
        dataUDP = new uchar[sizeUDP];
        memcpy(dataUDP, dataGlobal, sizeUDP);       
        
        int lengthToSend = sizeUDP;
        while(lengthToSend)
        {   
            step = stepUDP;
            if(lengthToSend < stepUDP)
                step = lengthToSend;

            send_buf = new uchar[step]; 
            memcpy(send_buf, dataUDP+startUDP, step);
            
            //cout<< hex << int(send_buf[0]) << " " << int(send_buf[1]) << " " << int(send_buf[2]) << " " << int(send_buf[3]) << endl;
            //char* pchr = (char*)(dataUDP+startUDP);
            //send_num = sendto(sock_fd, pchr, step, 0, (struct sockaddr *)&addr_serv, len);
            send_num = sendto(sock_fd, send_buf, step, 0, (struct sockaddr *)&addr_serv, len);
            //cout << "send num picture: " << send_num << endl;
            lengthToSend = lengthToSend - step;
            startUDP = startUDP + step;
            //cout << "send: " << step << " Bytes. " << "lengthToSend: " << lengthToSend << endl;
            delete send_buf;
            int fpsUDP = 1000*1000/(2*1024*1024/sizeUDP);
        }
        delete dataUDP;
        startUDP = 0;
        double timePoint2 = cv::getTickCount();
        //cout << "Send to UDP FPS:        " << cv::getTickFrequency()/(timePoint2-timePoint1) << endl;
        ResetEvent(encodePicureEvent); 
    }
    
    // test queue
    while (0)
    {
        double end = (cv::getTickCount() - start) *1000/(getTickFrequency());
        count ++ ; 
        if(end >1000)
        {
            cout << "--------------------count---------------------" << count << endl;
            start = cv::getTickCount();
            count = 0;
        }
        double timePoint1 = cv::getTickCount();
        if(queueData.empty())
            continue;
        
        vector<uchar> dataUDPVector = queueData.front();
        
        
        uchar *dataUDP = &dataUDPVector[0];
        sizeUDP = dataUDPVector.size();
               
        
        int lengthToSend = sizeUDP;
        while(lengthToSend)
        {   
            step = stepUDP;
            if(lengthToSend < stepUDP)
                step = lengthToSend;

            send_buf = new uchar[step]; 
            memcpy(send_buf, dataUDP+startUDP, step);
            
            //cout<< hex << int(send_buf[0]) << " " << int(send_buf[1]) << " " << int(send_buf[2]) << " " << int(send_buf[3]) << endl;
            //char* pchr = (char*)(dataUDP+startUDP);
            //send_num = sendto(sock_fd, pchr, step, 0, (struct sockaddr *)&addr_serv, len);
            send_num = sendto(sock_fd, send_buf, step, 0, (struct sockaddr *)&addr_serv, len);
            
            lengthToSend = lengthToSend - step;
            startUDP = startUDP + step;
            //cout << "send: " << step << " Bytes. " << "lengthToSend: " << lengthToSend << endl;
            delete send_buf;
            //usleep(20000);
            
        }
        queueData.pop();
        //delete dataUDP;
        startUDP = 0;
        double timePoint2 = cv::getTickCount();
        cout << "Send to UDP FPS:        ----------------------" << cv::getTickFrequency()/(timePoint2-timePoint1) << endl <<endl;
        ResetEvent(encodePicureEvent); 
    }
}


int main(int argc, char *argv[])
{
    struct ifreq ifreq;
    int sock = 0;
    char mac[32] = "";
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if(sock < 0)
    {
        perror("error sock");
        return 0;
    }
    strcpy(ifreq.ifr_name, ETH_NAME);
    if(ioctl(sock, SIOCGIFHWADDR, &ifreq) < 0)
    {
        perror("error ioctl");
        return 0;
    }
    
    int i = 0;
    for (i =0; i<6; i++)
    {
        sprintf(mac + 3*i, "%02X:", (unsigned char)ifreq.ifr_hwaddr.sa_data[i]);
    }
    mac[strlen(mac)-1] =0;
    //printf("MAC: %s \n", mac);
    char macaddr[32] = "48:B0:2D:35:9E:D5";
    //cout << mac << endl;
    if(strcmp(mac, macaddr)!=0)
    {
        perror("error device!");
        return 0; 
    }
    
    if(argc == 2)
    {
        //cout << (int)*argv[1] << endl;
        if((int)*argv[1] == 48)
            flag_show = false;
    }
        
	receivePictureEvent = CreateEvent(true, false);
    encodePicureEvent = CreateEvent(true, false);
    sendPictureEvent = CreateEvent(true, false);

	pthread_t pid1;
    pthread_t pid2;
	pthread_t pid3;
    pthread_t pid4;


    pthread_create(&pid1, NULL, readCamera, NULL); 
    pthread_create(&pid2, NULL, detection, NULL);
    pthread_create(&pid3, NULL, encodeImg, NULL);
    pthread_create(&pid4, NULL, sendUDP, NULL); 
	
	pthread_join(pid1, NULL); 
    pthread_join(pid2, NULL);
    pthread_join(pid3, NULL); 
    pthread_join(pid4, NULL);

	CloseHandle(receivePictureEvent);
	return 0;
}


