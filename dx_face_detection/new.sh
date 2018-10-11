#!/bin/bash

print_help() {
	echo ""
	echo "run_fd.sh - a wrapper script for face detection, age/gender recognition, and head pose estimation"
	echo ""
	echo "    Usage: run_fd.sh [-v] =x [-y] [=z]"
	echo ""
	echo "    [-h] Print this help"
	echo "    [-v] Optional - provide a full path to a video file"
	echo "    [-x] Select the hardware target for face detection."
	echo "    [-y] Optional - Requires -x.  Select the hardware target for age/gender recognition."
	echo "    [-z] Optional - Requires -x and -y.  Select the hardware target for head pose recognition."
	echo ""
	exit
}

while getopts ":h:v:x:y:z:" opt; do
  case ${opt} in
    h )
		print_help
      ;;
    v ) 
		vid=$OPTARG
      ;;
	x ) # Face Detection HW
		hwFd=$OPTARG
      ;;
	y ) # Age/Gender HW
		hwAg=$OPTARG
      ;;
	z ) # Head Pose HW
		hwHp=$OPTARG
      ;;
	* ) 
		print_help 
      ;;
  esac
done
shift $((OPTIND -1))


adjust_params() {
  hw="CPU"
  het="HETERO:FPGA,CPU"
  fp="FP32"
  fp_ag="FP32"
  fp_hp="FP32"

	if [[ $vid == "" ]]; then
		vid=../Videos/head-pose-face-detection-female-and-male.mp4 
	fi
	if [[ $hwFd != "" ]]; then
		hw="$hwFd"
	fi
	if [[ $hwAg != "" ]]; then
		hw_ag="$hwAg"
	fi
	if [[ $hwHp != "" ]]; then
		hw_hp="$hwHp"
	fi

  # Expand FPGA, Capitalize Other Strings
  if [[ $hw == "fpga" ]]; then
    hw="$het"
  else
    hw=${hw^^}
  fi
  if [[ $hw_ag == "fpga" ]]; then
    hw_ag="$het"
  else
    hw_ag=${hw_ag^^}
  fi
  if [[ $hw_hp == "fpga" ]]; then
		echo "Warning: FPGA doesn't support the head pose model - changing to CPU."
		hw_hp="CPU"
  else
    hw_hp=${hw_hp^^}
  fi

	# Update FP Settings
	if [[ $hw != "CPU" ]]; then
		fp="FP16"
	fi

	if [[ $hw_ag != "CPU" ]]; then
		fp_ag="FP16"
	fi

	if [[ $hw_hp != "CPU" ]]; then
		fp_hp="FP16"
	fi
}

# Main
#--------------------------------------------------------------------------------
if [[ $hwFd == "" ]]; then
	print_help
	exit
fi

adjust_params

fd=build/intel64/Release/face_detection_tutorial
moddir=/opt/intel/computer_vision_sdk/deployment_tools/intel_models
mod_od=$moddir/face-detection-retail-0004/$fp/face-detection-retail-0004.xml 
mod_ag=$moddir/age-gender-recognition-retail-0013/$fp_ag/age-gender-recognition-retail-0013.xml 
mod_hp=$moddir/head-pose-estimation-adas-0001/$fp_hp/head-pose-estimation-adas-0001.xml

# Execute Face Detection 
if [[ -n $hw_ag && -n $hw_hp ]]; then
		# face detection + age & gender + head pose
	$fd -i $vid -m $mod_od -m_ag $mod_ag -m_hp $mod_hp -d $hw -d_ag $hw_ag -d_hp $hw_hp
	echo "$fd -i $vid -m $mod_od -m_ag $mod_ag -m_hp $mod_hp -d $hw -d_ag $hw_ag -d_hp $hw_hp"
else
	if [[ -n $hw_ag ]]; then
		# face detection + age & gender
		$fd -i $vid -m $mod_od -m_ag $mod_ag -d $hw -d_ag $hw_ag
		echo "$fd -i $vid -m $mod_od -m_ag $mod_ag -d $hw -d_ag $hw_ag"
	else
		# Just face detection
		$fd -i $vid -m $mod_od -d $hw
		echo "$fd -i $vid -m $mod_od -d $hw"
	fi
fi

exit

