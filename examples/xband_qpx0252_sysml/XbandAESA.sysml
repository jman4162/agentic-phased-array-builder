package XbandAESA {
    public import RFVocabulary::*;
    public import RFParts::*;
    public import RFRequirements::*;
    public import RFAnalyses::*;
    doc /* 9.5 GHz search radar aperture: 32x32 elements, one QPF5012-class FEM per element, one 4-channel beamformer IC per 2x2 tile driving the FEMs directly. The beamformer's two receive modes are verified as separate studies against the same requirements. */
    part radar {
        part array : PhasedArrayAntenna {
            attribute frequency = 9.5 [GHz];
            attribute elementsX = 32;
            attribute elementsY = 32;
            part fem : TRModule {
                doc /* QPF5012-class GaN/GaAs FEM, 8.5-10.5 GHz: 10 W Psat, 23 dB Rx gain, 2.2 dB NF. One per element. */
                attribute count = 1024;
            }
            part beamformer : Beamformer {
                doc /* Four-channel analog beamformer IC with direct FEM drive and selectable receive modes. One per 2x2 tile. */
                attribute count = 256;
            }
        }
    }
    requirement <'REQ-RANGE'> DetectionRangeFloor {
        doc /* Detect a 1 m^2 Swerling 1 target at Pd 0.9, Pfa 1e-6, 16 pulses noncoherent, at 24 km or more. */
        attribute metricKey = "detection_range_m";
        attribute threshold = 24000.0;
        attribute op = ">=";
        attribute severity = "must";
    }
    requirement <'REQ-IIP3'> ReceiveLinearityFloor {
        doc /* Cascaded receive IIP3 of -13 dBm or more, which keeps a 25 kW marine navigation radar's sidelobe (79 dBm EIRP) below element 1 dB compression at 500 m. */
        attribute metricKey = "cascade_iip3_dbm";
        attribute threshold = -13.0;
        attribute op = ">=";
        attribute severity = "must";
    }
    analysis lowNoiseStudy : LinkBudgetAnalysis {
        subject radar;
        objective { doc /* Radar detection and receive cascade with the beamformer in its low-noise mode. */ }
    }
    metadata lowNoiseBinding : verificationBinding about lowNoiseStudy {
        engine = "phased-array-systems";
        configRef = "low_noise_mode.yaml";
    }
    analysis highLinearityStudy : LinkBudgetAnalysis {
        subject radar;
        objective { doc /* Radar detection and receive cascade with the beamformer in its high-linearity mode. */ }
    }
    metadata highLinearityBinding : verificationBinding about highLinearityStudy {
        engine = "phased-array-systems";
        configRef = "high_linearity_mode.yaml";
    }
    dependency verify_1 from lowNoiseStudy to DetectionRangeFloor;
    dependency verify_2 from lowNoiseStudy to ReceiveLinearityFloor;
    dependency verify_3 from highLinearityStudy to DetectionRangeFloor;
    dependency verify_4 from highLinearityStudy to ReceiveLinearityFloor;
    satisfy DetectionRangeFloor by radar.array;
    satisfy ReceiveLinearityFloor by radar.array;
}
