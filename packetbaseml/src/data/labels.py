cicids2018_labels = [
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

cicids2018_train_labels = [
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

cicids2018_test_labels = [
    label for label in cicids2018_labels if label not in cicids2018_train_labels
]


nb15v2_labels = [
    "Analysis",
    "Backdoor",
    "Benign",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

nb15v2_train_labels = [
    "Analysis",
    "Benign",
    "DoS",
    "Exploits",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
]

nb15v2_test_labels = [
    label for label in nb15v2_labels if label not in nb15v2_train_labels
]
