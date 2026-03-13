#pragma once
#include <string>

// ==============================================================================
// DISTRIBUTED COMMUNICATION (MOCK ALGORITHM)
// This class simulates sending target coordinates via UDP to Ground Station
// ==============================================================================
class TelemetrySender {
public:
    void sendDataToGroundStation(int target_x, int target_y, bool is_locked) {
        // In a real scenario, this would serialize data to JSON and send via UDP socket
        /*
         std::string payload = "{\"x\": " + std::to_string(target_x) + 
                               ", \"y\": " + std::to_string(target_y) + 
                               ", \"locked\": " + std::to_string(is_locked) + "}";
         udp_socket.send(payload);
        */
    }
};