"use client";


import { useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import ResultCard from "../../../components/ResultCard";
import Loader from '@/components/Loader';
import { Debate } from "@/types/debate";

const Result = () => {
    const params = useParams<{ id: string }>();
    const id = params.id;

    const [debate, setDebate] = useState<Debate | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() =>{
        const fetchData = async() => {
            const res = await fetch(`/api/debate/${id}`);
            const data = await res.json();

            setDebate(data);
            setLoading(false);
        }
        fetchData();
    },[id]);
    console.log(debate);

    if (loading || !debate) {
  return <Loader />;
}

  return (
    <>
        <h1>Result</h1>
        <ResultCard debate={debate}/>
    </>
  )
}

export default Result;