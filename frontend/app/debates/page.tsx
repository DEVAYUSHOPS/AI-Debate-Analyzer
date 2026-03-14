"use client";

import React, { useEffect,useState } from 'react'
import { Debate } from "@/types/debate";
import AllDebates from '@/components/AllDebates';
import PageContainer from '@/components/PageContainer';

const Debates = () => {
    const [debates, setDebates] = useState<Debate[]>([]);
    useEffect(() =>{    
        const fetchDebates = async() =>{
            const res = await fetch('/api/debates');
            const data = await res.json();
            setDebates(data);
            console.log(data)
        }
        fetchDebates();
    },[])
  return (
    <>
        <PageContainer>
            <h1>Debate History</h1>
            <AllDebates debates={debates} />
        </PageContainer>
    </>
  )
}

export default Debates;