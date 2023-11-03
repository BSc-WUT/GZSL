labels = [
    "Benign",
    "Bot",
    "Brute Force -XSS",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DDoS attacks-LOIC-HTTP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-Slowloris",
    "FTP-BruteForce",
    "SSH-Bruteforce",
    "Label",
    "Brute Force -Web",
    "DoS attacks-SlowHTTPTest",
    "Infilteration",
    "SQL Injection",
]

train_labels = [
    "Bot",
    "Brute Force -XSS",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DDoS attacks-LOIC-HTTP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-Slowloris",
    "FTP-BruteForce",
    "SSH-Bruteforce",
    "Label",
]

test_labels = [label for label in labels if label not in train_labels]
