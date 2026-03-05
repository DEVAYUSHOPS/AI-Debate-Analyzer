"use client";

import React, { useState } from 'react';
import { useRouter } from "next/navigation";
import Loader from './Loader';

const DebateForm = () => {
    
    const router = useRouter();
    const [topic, setTopic] = useState("");
    const [speakerA, setSpeakerA] = useState("");
    const [speakerB, setSpeakerB] = useState("");
    const [transcript, setTranscript] = useState("");
    const [loading,setLoading] = useState(false);
    if(loading){
        return <Loader />;
    }
    const handleSubmit = async(e: React.FormEvent<HTMLFormElement>) => {
        if (!topic || !speakerA || !speakerB || !transcript){
            alert("Please fill all fields");
            return;
        }
        e.preventDefault();
        try {
            setLoading(true);
        const res = await fetch("/api/analyze",{
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body:JSON.stringify({
                topic,
                speakerA,
                speakerB,
                transcript
            })
        })
        setLoading(false);
        const data = await res.json();
        console.log(data);

        router.push(`/result/${data.debateId}`)
        } catch (error) {
            console.log(error);
        }
        
    }



  return (
    <>
        <form onSubmit={handleSubmit}>
            <h1>Topic: </h1>
            <input type="text" 
            placeholder="Enter debate topic"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            required
            />
            <h1>Speaker A: </h1>
            <input type="text" 
            placeholder="Enter name of Speaker A"
            value={speakerA}
            onChange={(e) => setSpeakerA(e.target.value)}
            required
            />
            <h1>Speaker B: </h1>
            <input type="text" 
            placeholder="Enter name of Speaker B"
            value={speakerB}
            onChange={(e) => setSpeakerB(e.target.value)}
            required
            />
            <h1>Transcript: </h1>
            <textarea 
            placeholder="Paste debate transcript here"
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            required
            />
            <button type="submit">Analyze Debate</button>
        </form>
    </>
  )
}

export default DebateForm