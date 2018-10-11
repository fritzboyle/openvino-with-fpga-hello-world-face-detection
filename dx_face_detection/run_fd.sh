#!/bin/bash

get_params() {
  hw="CPU"
  het="HETERO:FPGA,CPU"
  solo_mode=1
	fp="FP32"
	fp_ag="FP32"
	fp_hp="FP32"

  # Get Hardware Target Parameters 
  if [[ -n $@ ]]; then
    if [[ -n $1 ]]; then
      hw="$1"
    fi
    if [[ -n $2 ]]; then
      hw_ag="$2"
      solo_mode=0
    fi
    if [[ -n $3 ]]; then
      hw_hp="$3"
      solo_mode=0
    fi
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
get_params $@

fd=build/intel64/Release/face_detection_tutorial
vid=../Videos/head-pose-face-detection-female-and-male.mp4 
moddir=/opt/intel/computer_vision_sdk/deployment_tools/intel_models
mod_od=$moddir/face-detection-retail-0004/$fp/face-detection-retail-0004.xml 
mod_ag=$moddir/age-gender-recognition-retail-0013/$fp_ag/age-gender-recognition-retail-0013.xml 
mod_hp=$moddir/head-pose-estimation-adas-0001/$fp_hp/head-pose-estimation-adas-0001.xml


# Execute Face Detection 
if [[ -n $hw_ag && -n $hw_hp ]]; then
		# face detection + age & gender + head pose
	$fd -i $vid -m $mod_od -m_ag $mod_ag -m_hp $mod_hp -d $hw -d_ag $hw_ag -d_hp $hw_hp
else
	if [[ -n $hw_ag ]]; then
		# face detection + age & gender
		$fd -i $vid -m $mod_od -m_ag $mod_ag -d $hw -d_ag $hw_ag
	else
		# Just face detection
		$fd -i $vid -m $mod_od -d $hw
	fi
fi

exit

